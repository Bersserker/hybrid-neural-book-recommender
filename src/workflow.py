"""Автономный сквозной пайплайн из prototype.ipynb."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.config import PROJECT_DIR as PROJECT_ROOT
from src.data.features import build_feature_tables
from src.data.load_dataset import laod_dataframes
from src.data.prepare_data import clean_ratings, train_test_val_per_user
from src.metrics.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.metrics.segments import SEGMENT_ORDER, make_user_segments, summarize_segments
from src.models.adaptive_hybrid_recommender import (
    AdaptiveHybridRecommender,
    optimize_adaptive_weights,
)
from src.models.balanced_recommender import (
    BalancedRecommender,
    intra_list_diversity,
    tune_mmr,
)
from src.models.cb_item_recomennder import cb_item_recomennder
from src.models.content_based_recomender import content_based_recomender
from src.models.hybrid_recommender import HybridRecommender
from src.models.recommend_popular_recomemder import recommend_popular_recomemder
from src.models.svd_recommender import svd_recommender
from src.models.two_tower import TwoTowerRecommender


def _sample_relevant(relevant, max_users, random_state):
    if max_users is None or len(relevant) <= max_users:
        return dict(relevant)
    users = np.random.default_rng(random_state).choice(
        sorted(relevant), max_users, replace=False
    )
    return {int(u): relevant[u] for u in users}


def evaluate_pipeline_models(recommenders, relevant, content_model, k):
    """Один вызов модели на пользователя: качество и ILD считаются на одной выдаче."""
    per_user, summary = {}, []
    for name, recommend in recommenders.items():
        rows = []
        for user_id, actual in relevant.items():
            recs = recommend(user_id, N=k)
            predicted = [rec["book_id"] for rec in recs]
            rows.append(
                {
                    "user_id": user_id,
                    "precision": precision_at_k(actual, predicted, k),
                    "recall": recall_at_k(actual, predicted, k),
                    "ndcg": ndcg_at_k(actual, predicted, k),
                    "ild": intra_list_diversity(recs, content_model),
                }
            )
        frame = pd.DataFrame(rows)
        per_user[name] = frame
        summary.append(
            {
                "Model": name,
                "users": len(frame),
                f"Precision@{k}": frame["precision"].mean(),
                f"Recall@{k}": frame["recall"].mean(),
                f"nDCG@{k}": frame["ndcg"].mean(),
                f"ILD@{k}": frame["ild"].mean(),
            }
        )
    return pd.DataFrame(summary).sort_values(f"nDCG@{k}", ascending=False), per_user


def run_pipeline(
    data_dir=PROJECT_ROOT / "data" / "raw",
    output_dir=PROJECT_ROOT / "artifacts" / "notebook_pipeline",
    download=False,
    max_users=None,
    use_neural=True,
    epochs=5,
    batch_size=1024,
    svd_epochs=50,
    n_trials=30,
    tuning_users=200,
    eval_users=None,
    k=10,
    candidates_k=100,
    recommend_user_ids=None,
    random_state=42,
    device=None,
):
    """CSV → split → признаки → обучение → подбор на val → выдача → test-отчёт.

    Все модели создаются заново и используют локальные данные этого запуска.
    Результат содержит готовую модель и таблицы; повторное обучение для выдачи не нужно.
    """
    if k < 1 or candidates_k < k or n_trials < 0 or svd_epochs < 1:
        raise ValueError(
            "Нужны k >= 1, candidates_k >= k, svd_epochs >= 1, n_trials >= 0"
        )
    if use_neural and (epochs < 1 or batch_size < 1):
        raise ValueError("Для Two Tower нужны положительные epochs и batch_size")
    for limit in [max_users, tuning_users, eval_users]:
        if limit is not None and limit < 1:
            raise ValueError(
                "Ограничение пользователей должно быть положительным или None"
            )
    started = perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if download:
        # Скачивание включается явно; готовые локальные CSV скачивать заново не нужно.
        from src.data.make_dataset import load_kaggle_dataset

        load_kaggle_dataset(path=data_dir)

    print("1/6 Загрузка и подготовка данных", flush=True)
    data = laod_dataframes(data_dir)
    books, book_tags, tags = data["df_books"], data["df_book_tags"], data["df_tags"]
    cleaned = clean_ratings(data["df_ratings"], books)
    if max_users is not None and cleaned["user_id"].nunique() > max_users:
        chosen = np.random.default_rng(random_state).choice(
            np.sort(cleaned["user_id"].unique()), max_users, replace=False
        )
        cleaned = cleaned[cleaned["user_id"].isin(chosen)].reset_index(drop=True)
    train, validation, test = train_test_val_per_user(
        cleaned, random_state=random_state
    )

    def relevant_from(frame):
        return (
            frame[frame["rating"] >= 4].groupby("user_id")["book_id"].agg(set).to_dict()
        )

    val_relevant_local, test_relevant_local = (
        relevant_from(validation),
        relevant_from(test),
    )
    if not val_relevant_local or not test_relevant_local:
        raise ValueError(
            "Нужны релевантные книги в val и test; увеличьте выборку пользователей"
        )
    tuning_relevant = _sample_relevant(val_relevant_local, tuning_users, random_state)
    evaluation_relevant = _sample_relevant(
        test_relevant_local, eval_users, random_state
    )
    builder, user_table, book_table = build_feature_tables(
        train, books, book_tags, tags
    )
    train_table = builder.transform(train, leave_one_out=True)

    print("2/6 Обучение классических моделей и Two Tower", flush=True)
    base = {}
    training_times = []
    factories = {
        "pop": lambda: recommend_popular_recomemder(train),
        "content": lambda: content_based_recomender(train, book_tags, books, tags),
        "item": lambda: cb_item_recomennder(train, books),
        "svd": lambda: svd_recommender(
            train, books, n_epochs=svd_epochs, random_state=random_state
        ),
    }
    for name, factory in factories.items():
        timer = perf_counter()
        base[name] = factory()
        training_times.append({"model": name, "seconds": perf_counter() - timer})
    if use_neural:
        timer = perf_counter()
        base["two_tower"] = TwoTowerRecommender(
            train,
            books,
            train_features=train_table,
            user_features=user_table,
            book_features=book_table,
            random_state=random_state,
            device=device,
        ).fit(tuning_relevant, epochs=epochs, batch_size=batch_size, k=k)
        training_times.append({"model": "two_tower", "seconds": perf_counter() - timer})

    print("3/6 Подбор весов гибрида и MMR только на validation", flush=True)
    classical = HybridRecommender(
        train,
        models={name: model for name, model in base.items() if name != "two_tower"},
    )
    hybrid = HybridRecommender(train, models=base)
    initial_adaptive = AdaptiveHybridRecommender(train, models=base)
    adaptive = AdaptiveHybridRecommender(train, models=base)
    search = pd.DataFrame(columns=["user_type", "trial", *base, f"nDCG@{k}"])
    if n_trials:
        adaptive.adaptive_weights, search = optimize_adaptive_weights(
            adaptive,
            tuning_relevant,
            n_trials=n_trials,
            k=k,
            candidates_k=candidates_k,
            random_state=random_state,
        )
    mmr_weight, mmr_table, _ = tune_mmr(
        adaptive,
        base["content"],
        tuning_relevant,
        k=k,
        candidates_k=candidates_k,
        max_users=None,
        random_state=random_state,
    )
    final_model = BalancedRecommender(
        adaptive, base["content"], mmr_weight, candidates_k
    )

    print("4/6 Генерация итоговых рекомендаций", flush=True)
    if recommend_user_ids is None:
        recommend_user_ids = sorted(evaluation_relevant)[:5]
    recommendation_rows = []
    for user_id in recommend_user_ids:
        for rank, rec in enumerate(final_model.recommend(user_id, N=k), start=1):
            recommendation_rows.append({"user_id": user_id, "rank": rank, **rec})
    recommendations = pd.DataFrame(
        recommendation_rows,
        columns=["user_id", "rank", "book_id", "score", "mmr_score"],
    )
    recommendations = recommendations.merge(
        books[["id", "title", "authors"]].rename(columns={"id": "book_id"}),
        on="book_id",
        how="left",
        validate="many_to_one",
    )

    print("5/6 Итоговая оценка на test и анализ сегментов", flush=True)
    labels = {
        "pop": "Popularity",
        "content": "Content-Based",
        "item": "Item-Based CF",
        "svd": "SVD",
        "two_tower": "Two Tower",
    }
    recommenders = {labels[name]: model.recommend for name, model in base.items()}
    recommenders["Hybrid (classical)"] = classical.recommend
    if use_neural:
        recommenders["Hybrid + Two Tower"] = hybrid.recommend
    recommenders["Adaptive Hybrid"] = initial_adaptive.recommend
    recommenders["Adaptive Hybrid (tuned)"] = adaptive.recommend
    recommenders["Adaptive Hybrid + MMR"] = final_model.recommend
    metrics, user_metrics = evaluate_pipeline_models(
        recommenders, evaluation_relevant, base["content"], k
    )
    segments, population = make_user_segments(train, evaluation_relevant)
    segment_metrics = summarize_segments(user_metrics, segments, k=k)

    print("6/6 Сохранение результатов и отчёта", flush=True)
    processed_dir, report_dir = output_dir / "processed", output_dir / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train), ("val", validation), ("test", test)]:
        frame.to_csv(processed_dir / f"{name}.csv", index=False)
    metrics.to_csv(report_dir / "metrics.csv", index=False)
    segment_metrics.to_csv(report_dir / "segments.csv", index=False)
    population.to_csv(report_dir / "segment_population.csv")
    recommendations.to_csv(report_dir / "recommendations.csv", index=False)
    mmr_table.to_csv(report_dir / "mmr_validation.csv", index=False)
    search.to_csv(report_dir / "weight_search.csv", index=False)
    pd.DataFrame(training_times).to_csv(report_dir / "training_times.csv", index=False)
    for name, frame in user_metrics.items():
        frame.to_csv(report_dir / f"users_{name.replace(' ', '_')}.csv", index=False)
    if use_neural:
        base["two_tower"].history.to_csv(
            report_dir / "two_tower_history.csv", index=False
        )
    config = {
        "data_dir": str(data_dir.resolve()),
        "random_state": random_state,
        "max_users": max_users,
        "use_neural": use_neural,
        "epochs": epochs,
        "batch_size": batch_size,
        "svd_epochs": svd_epochs,
        "n_trials": n_trials,
        "k": k,
        "candidates_k": candidates_k,
        "tuning_users_limit": tuning_users,
        "tuning_users": len(tuning_relevant),
        "eval_users_limit": eval_users,
        "evaluated_test_users": len(evaluation_relevant),
        "all_eligible_test_users": len(test_relevant_local),
        "train_rows": len(train),
        "val_rows": len(validation),
        "test_rows": len(test),
        "model_names": list(base),
        "mmr_weight": mmr_weight,
        "duration_seconds": perf_counter() - started,
    }
    (report_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "adaptive_weights.json").write_text(
        json.dumps(adaptive.adaptive_weights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ax = metrics.set_index("Model")[
        [f"Precision@{k}", f"Recall@{k}", f"nDCG@{k}"]
    ].plot.bar(figsize=(13, 5), rot=25)
    ax.set_title("Test metrics")
    ax.figure.tight_layout()
    ax.figure.savefig(report_dir / "metrics.png", dpi=150)
    plt.close(ax.figure)
    ax = (
        segment_metrics.pivot(index="Segment", columns="Model", values=f"nDCG@{k}")
        .reindex(SEGMENT_ORDER)
        .plot.bar(figsize=(13, 5), rot=0)
    )
    ax.set_title("Test nDCG by train activity")
    ax.figure.tight_layout()
    ax.figure.savefig(report_dir / "segments.png", dpi=150)
    plt.close(ax.figure)
    report_path = report_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Сквозной запуск рекомендательной системы",
                "",
                f"Источник: `{data_dir.resolve()}`; seed={random_state}; K={k}.",
                "",
                f"Test-пользователей: {len(evaluation_relevant)} из {len(test_relevant_local)} с релевантными книгами.",
                "Веса моделей, баланс MMR и лучшая эпоха выбраны только по validation.",
                f"Модели: {', '.join(base)}. Вес релевантности MMR: {mmr_weight}.",
                "",
                "## Сравнение на test",
                "",
                "```text",
                metrics.to_string(index=False),
                "```",
                "",
                "![Метрики](metrics.png)",
                "",
                "## Сегменты",
                "",
                "```text",
                population.to_string(),
                "```",
                "",
                "```text",
                segment_metrics.to_string(index=False),
                "```",
                "",
                "![Сегменты](segments.png)",
                "",
                "## Примеры рекомендаций",
                "",
                "```text",
                recommendations.head(30).to_string(index=False),
                "```",
                "",
                "Метрики — средние по пользователям. Сегменты определены числом train-оценок.",
                "Сравните Hybrid (classical) и Hybrid + Two Tower, затем адаптивный гибрид и MMR.",
                "Превосходство одной модели не предполагается заранее; результаты требуют интерпретации.",
                "Полные параметры и размеры выборок сохранены в config.json.",
            ]
        ),
        encoding="utf-8",
    )
    print("Отчёт:", report_path)
    return {
        "model": final_model,
        "base_models": base,
        "hybrid": hybrid,
        "adaptive": adaptive,
        "features": builder,
        "metrics": metrics,
        "segments": segment_metrics,
        "recommendations": recommendations,
        "report_path": report_path,
        "config": config,
    }
