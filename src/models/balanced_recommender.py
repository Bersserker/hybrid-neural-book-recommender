"""MMR: баланс релевантности и разнообразия, настройка только на validation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.metrics.metrics import ndcg_at_k, precision_at_k, recall_at_k


def book_similarity(recommendations, content_model):
    """Сходство только книг-кандидатов: маленькая матрица вместо всего каталога."""
    indices = [content_model.book_id_to_idx[rec["book_id"]] for rec in recommendations]
    vectors = content_model.tfidf_matrix[indices]
    return cosine_similarity(vectors).clip(0, 1)


def mmr_rerank(recommendations, content_model, N=10, relevance_weight=0.8):
    """Выбрать N книг, балансируя score и сходство с уже выбранными книгами."""
    if not 0 <= relevance_weight <= 1:
        raise ValueError("relevance_weight должен быть от 0 до 1")
    if N <= 0 or not recommendations:
        return []

    # Начинаем с наиболее релевантных, заодно убираем повторные book_id.
    candidates = sorted(
        recommendations, key=lambda rec: (-rec["score"], rec["book_id"])
    )
    candidates = pd.DataFrame(candidates).drop_duplicates("book_id").to_dict("records")
    scores = np.array([rec["score"] for rec in candidates], dtype=float)
    # Релевантность и cosine similarity должны иметь одинаковую шкалу [0, 1].
    if np.isclose(scores.max(), scores.min()):
        relevance = np.ones(len(scores))
    else:
        relevance = (scores - scores.min()) / (scores.max() - scores.min())
    similarity = book_similarity(candidates, content_model)

    selected = []
    remaining = list(range(len(candidates)))
    result = []
    for _ in range(min(N, len(candidates))):
        best_index = None
        best_score = -np.inf
        for index in remaining:
            # Насколько кандидат похож на самую похожую уже выбранную книгу?
            redundancy = max(similarity[index, j] for j in selected) if selected else 0
            mmr_score = (
                relevance_weight * relevance[index]
                - (1 - relevance_weight) * redundancy
            )
            if mmr_score > best_score:
                best_index, best_score = index, mmr_score
        selected.append(best_index)
        remaining.remove(best_index)
        # score оставляем исходным; mmr_score объясняет выбор на этом шаге.
        result.append({**candidates[best_index], "mmr_score": float(best_score)})
    return result


def intra_list_diversity(recommendations, content_model):
    """ILD: среднее расстояние между всеми разными парами книг; больше — разнообразнее."""
    if len(recommendations) < 2:
        return 0.0
    similarity = book_similarity(recommendations, content_model)
    upper_triangle = np.triu_indices(len(recommendations), k=1)
    return float((1 - similarity[upper_triangle]).mean())


def tune_mmr(
    model,
    content_model,
    relevant,
    k=10,
    candidates_k=100,
    max_users=200,
    max_ndcg_drop=0.02,
    random_state=42,
):
    """Выбрать баланс только по validation; тестовые оценки сюда не передаются."""
    if not relevant or k < 1 or candidates_k < 1:
        raise ValueError("Нужны непустая validation и положительные k/candidates_k")
    if not 0 <= max_ndcg_drop <= 1 or (max_users is not None and max_users < 1):
        raise ValueError("Недопустимый допуск или размер validation-подвыборки")
    users = sorted(relevant)
    if max_users is not None and len(users) > max_users:
        users = (
            np.random.default_rng(random_state)
            .choice(users, max_users, replace=False)
            .tolist()
        )
    candidates_k = max(candidates_k, k)
    candidates = {
        u: model.recommend(u, N=candidates_k, candidates_k=candidates_k) for u in users
    }
    results = []
    ndcg_column, ild_column = f"nDCG@{k}", f"ILD@{k}"
    for weight in [1.0, 0.9, 0.8, 0.7, 0.5]:
        rows = []
        for user_id in users:
            recs = mmr_rerank(
                candidates[user_id], content_model, N=k, relevance_weight=weight
            )
            predicted = [rec["book_id"] for rec in recs]
            actual = relevant[user_id]
            rows.append(
                {
                    f"Precision@{k}": precision_at_k(actual, predicted, k),
                    f"Recall@{k}": recall_at_k(actual, predicted, k),
                    ndcg_column: ndcg_at_k(actual, predicted, k),
                    ild_column: intra_list_diversity(recs, content_model),
                }
            )
        results.append(
            {"relevance_weight": weight, **pd.DataFrame(rows).mean().to_dict()}
        )
    comparison = pd.DataFrame(results)
    baseline = comparison.loc[comparison["relevance_weight"] == 1, ndcg_column].iloc[0]
    comparison["acceptable_ndcg"] = (
        comparison[ndcg_column] >= baseline * (1 - max_ndcg_drop) - 1e-12
    )
    best = (
        comparison[comparison["acceptable_ndcg"]]
        .sort_values([ild_column, ndcg_column, "relevance_weight"], ascending=False)
        .iloc[0]
    )
    return float(best["relevance_weight"]), comparison, candidates


class BalancedRecommender:
    """Хранит модель и выбранный вес: не зависит от глобального adaptive_model."""

    def __init__(self, model, content_model, relevance_weight, candidates_k=100):
        self.model = model
        self.content_model = content_model
        self.relevance_weight = relevance_weight
        self.candidates_k = candidates_k

    def recommend(self, user_id, N=10):
        if N <= 0:
            return []
        size = max(self.candidates_k, N)
        candidates = self.model.recommend(user_id, N=size, candidates_k=size)
        return mmr_rerank(
            candidates, self.content_model, N=N, relevance_weight=self.relevance_weight
        )


def train_balanced_recommender(
    processed_data_dir=None,
    raw_data_dir=None,
    model_path=None,
    model=None,
    n_trials=30,
    epochs=5,
    device=None,
    k=10,
    candidates_k=100,
):
    """Обучить адаптивный гибрид и подобрать MMR только по validation."""
    from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
    from src.models._io import MODELS_DIR, load_validation, save_model
    from src.models.adaptive_hybrid_recommender import train_adaptive_hybrid_recommender

    processed_data_dir = (
        PROCESSED_DATA_DIR if processed_data_dir is None else processed_data_dir
    )
    raw_data_dir = RAW_DATA_DIR if raw_data_dir is None else raw_data_dir
    model_path = (
        MODELS_DIR / "balanced_recommender.joblib" if model_path is None else model_path
    )
    if model is None:
        model = train_adaptive_hybrid_recommender(
            processed_data_dir,
            raw_data_dir,
            model_path=Path(model_path).with_name("adaptive_hybrid_recommender.joblib"),
            n_trials=n_trials,
            epochs=epochs,
            device=device,
            k=k,
            candidates_k=candidates_k,
        )
    content_model = model.models["content"]
    weight, comparison, _ = tune_mmr(
        model,
        content_model,
        load_validation(processed_data_dir),
        k=k,
        candidates_k=candidates_k,
        max_users=None,
    )
    balanced = BalancedRecommender(model, content_model, weight, candidates_k)
    balanced.search_results = comparison
    return save_model(balanced, model_path)


if __name__ == "__main__":
    from src.models.balanced_recommender import train_balanced_recommender

    train_balanced_recommender()
