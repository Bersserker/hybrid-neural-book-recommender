"""Отчёты о сохранённых моделях: ranking-метрики и история обучения."""

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_DIR
from src.metrics.evaluation import evaluate_model
from src.models._io import MODELS_DIR
from src.models.train import MODEL_FILES

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _markdown_table(frame):
    """Таблица Markdown без дополнительной зависимости tabulate."""

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.4f}" if np.isfinite(value) else "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    rows.extend(
        "| " + " | ".join(map(format_value, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def create_report(
    model="all",
    split="val",
    k=10,
    processed_data_dir=PROCESSED_DATA_DIR,
    models_dir=MODELS_DIR,
    reports_dir=PROJECT_DIR / "reports",
):
    """Оценить выбранные артефакты без обучения и вернуть пути к отчётам.

    all требует наличия всех моделей. Результаты разных выборок и наборов
    моделей сохраняются отдельно: reports/<split>/<model>/.
    """
    if model not in {*MODEL_FILES, "all"}:
        raise ValueError(f"Неизвестная модель: {model}")
    if split not in {"val", "test"} or k < 1:
        raise ValueError("Нужны split=val/test и k >= 1")
    models_dir = Path(models_dir)
    selected = list(MODEL_FILES) if model == "all" else [model]
    missing = [
        str(models_dir / MODEL_FILES[name])
        for name in selected
        if not (models_dir / MODEL_FILES[name]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Сначала обучите модели. Отсутствуют: " + ", ".join(missing)
        )
    with (Path(processed_data_dir) / f"{split}_relevant.pkl").open("rb") as file:
        relevant = pickle.load(file)
    if not relevant or any(not actual for actual in relevant.values()):
        raise ValueError("Для отчёта нужна непустая выборка с релевантными книгами")

    output_dir = Path(reports_dir) / split / model
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    paths = {}
    for name in selected:
        print(f"Оценка {name} на {split}: {len(relevant)} пользователей", flush=True)
        path = models_dir / MODEL_FILES[name]
        trained = joblib.load(path)
        started = perf_counter()
        per_user = evaluate_model(trained.recommend, relevant, k=k, verbose=True)
        evaluation_seconds = perf_counter() - started
        user_path = output_dir / f"{name}_users.csv"
        per_user.to_csv(user_path, index=False)
        paths[f"{name}_users"] = user_path
        metadata_path = path.with_suffix(".training.json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        rows.append(
            {
                "model": name,
                "users": len(per_user),
                f"Precision@{k}": per_user["precision"].mean(),
                f"Recall@{k}": per_user["recall"].mean(),
                f"nDCG@{k}": per_user["ndcg"].mean(),
                "training_seconds": metadata.get("duration_seconds", float("nan")),
                "evaluation_seconds": evaluation_seconds,
                "model_size_mb": path.stat().st_size / 1024**2,
            }
        )
        details.extend([f"## {name}", "", f"Артефакт: `{path.resolve()}`.", ""])
        if metadata:
            details.extend([f"Начало обучения (UTC): {metadata['started_at']}.", ""])
        if name == "two_tower" and hasattr(trained, "history"):
            history = trained.history
            history_path = output_dir / "two_tower_history.csv"
            history.to_csv(history_path, index=False)
            paths["two_tower_history"] = history_path
            details.extend(
                [
                    f"Лучшая эпоха по validation: {trained.best_epoch}.",
                    "",
                    _markdown_table(history),
                    "",
                ]
            )
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            history.plot(x="epoch", y="loss", ax=axes[0], title="Two Tower: train loss")
            score_column = next(c for c in history.columns if c.startswith("val_nDCG@"))
            history.plot(
                x="epoch", y=score_column, ax=axes[1], title="Two Tower: validation"
            )
            for ax in axes:
                ax.axvline(trained.best_epoch, color="gray", linestyle="--")
            fig.tight_layout()
            plot_path = output_dir / "two_tower_training.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            paths["two_tower_plot"] = plot_path
            details.extend(["![Обучение Two Tower](two_tower_training.png)", ""])
        if name == "adaptive":
            weights = pd.DataFrame.from_dict(trained.adaptive_weights, orient="index")
            weights.index.name = "user_type"
            weights_path = output_dir / "adaptive_weights.csv"
            weights.to_csv(weights_path)
            paths["adaptive_weights"] = weights_path
            details.extend(
                [
                    "Веса по группам пользователей:",
                    "",
                    _markdown_table(weights.reset_index()),
                    "",
                ]
            )
            if hasattr(trained, "search_results"):
                search_path = output_dir / "adaptive_weight_search.csv"
                trained.search_results.to_csv(search_path, index=False)
                paths["adaptive_search"] = search_path
        if name == "balanced":
            search_path = output_dir / "mmr_validation.csv"
            trained.search_results.to_csv(search_path, index=False)
            paths["mmr_search"] = search_path
            details.extend(
                [
                    f"Вес релевантности MMR, выбранный на validation: {trained.relevance_weight}.",
                    "",
                    _markdown_table(trained.search_results),
                    "",
                ]
            )

    summary = pd.DataFrame(rows).sort_values(
        f"nDCG@{k}", ascending=False, kind="stable"
    )
    metrics_path = output_dir / "metrics.csv"
    summary.to_csv(metrics_path, index=False)
    paths["metrics"] = metrics_path
    ax = summary.set_index("model")[
        [f"Precision@{k}", f"Recall@{k}", f"nDCG@{k}"]
    ].plot.bar(figsize=(10, 5), rot=25, title=f"Ranking metrics: {split}, K={k}")
    ax.set_ylim(0, 1)
    ax.figure.tight_layout()
    plot_path = output_dir / "metrics.png"
    ax.figure.savefig(plot_path, dpi=150)
    plt.close(ax.figure)
    paths["metrics_plot"] = plot_path
    report_path = output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Отчёт об обучении и оценке моделей",
                "",
                f"Выборка: **{split}**, K={k}, пользователей с релевантными книгами: {len(relevant)}.",
                "",
                "Метрики усреднены по всем пользователям выбранной выборки; релевантность — оценка ≥ 4.",
                "",
                "Время обучения включает сохранение артефакта. Для старых артефактов без метаданных оно не указано.",
                "",
                "## Сравнение моделей",
                "",
                _markdown_table(summary),
                "",
                "![Сравнение метрик](metrics.png)",
                "",
                *details,
            ]
        ),
        encoding="utf-8",
    )
    paths["report"] = report_path
    print(f"Отчёт сохранён: {report_path}", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODEL_FILES], default="all")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--processed-data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_DIR / "reports")
    create_report(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
