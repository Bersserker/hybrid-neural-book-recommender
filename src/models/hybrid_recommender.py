"""Взвешенное объединение классических моделей и Two Tower."""

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.models._io import MODELS_DIR, load_training_data, load_validation, save_model
from src.models.cb_item_recomennder import cb_item_recomennder
from src.models.content_based_recomender import content_based_recomender
from src.models.recommend_popular_recomemder import recommend_popular_recomemder
from src.models.svd_recommender import svd_recommender


def build_base_models(train_df, df_books, df_book_tags, df_tags, svd_kwargs=None):
    return {
        "pop": recommend_popular_recomemder(train_df),
        "content": content_based_recomender(train_df, df_book_tags, df_books, df_tags),
        "item": cb_item_recomennder(train_df, df_books),
        "svd": svd_recommender(train_df, df_books, **(svd_kwargs or {})),
    }


DEFAULT_WEIGHTS = {
    "pop": 0.08,
    "content": 0.16,
    "item": 0.24,
    "svd": 0.32,
    "two_tower": 0.20,
}


class HybridRecommender:
    def __init__(self, train_df, models):
        # Все модели уже обучены: здесь только объединяем их рекомендации.
        self.models = dict(models)
        if not self.models or not set(self.models).issubset(DEFAULT_WEIGHTS):
            raise ValueError("Нужны модели с ключами pop/content/item/svd/two_tower")
        total = sum(DEFAULT_WEIGHTS[name] for name in self.models)
        self.default_weights = {
            name: DEFAULT_WEIGHTS[name] / total for name in self.models
        }
        self.user_history = train_df.groupby("user_id")["book_id"].agg(set).to_dict()

    def normalize_scores(self, recommendations):
        if not recommendations:
            return {}

        df = pd.DataFrame(recommendations)

        min_score = df["score"].min()
        max_score = df["score"].max()

        if np.isclose(min_score, max_score):
            df["normalized_score"] = 1.0
        else:
            df["normalized_score"] = (df["score"] - min_score) / (max_score - min_score)

        return dict(zip(df["book_id"], df["normalized_score"]))

    def recommend(self, user_id, N=10, weights=None, candidates_k=100):
        if N <= 0:
            return []
        if candidates_k < 1:
            raise ValueError("candidates_k должен быть положительным")
        candidates_k = max(candidates_k, N)
        weights = dict(self.default_weights if weights is None else weights)

        if set(weights) != set(self.models):
            raise ValueError(f"Веса должны содержать ключи: {list(self.models)}")

        if any(not np.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Веса должны быть конечными неотрицательными числами")

        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Сумма весов должна равняться 1")

        # Получаем рекомендации базовых моделей
        recommendations = {
            name: model.recommend(user_id, N=candidates_k)
            for name, model in self.models.items()
        }

        # Используем существующую функцию normalize_scores:
        # она должна возвращать словарь {book_id: normalized_score}
        normalized = {
            name: self.normalize_scores(recs) for name, recs in recommendations.items()
        }

        candidate_ids = set().union(*(scores.keys() for scores in normalized.values()))

        # Исключаем прочитанные книги
        seen = self.user_history.get(user_id, set())
        candidate_ids -= seen

        results = []

        for book_id in candidate_ids:
            score = sum(
                weights[name] * normalized[name].get(book_id, 0.0)
                for name in self.models
            )

            results.append(
                {
                    "book_id": int(book_id),
                    "score": float(score),
                }
            )

        results.sort(key=lambda rec: (-rec["score"], rec["book_id"]))

        return results[:N]


def train_hybrid_recommender(
    processed_data_dir=PROCESSED_DATA_DIR,
    raw_data_dir=RAW_DATA_DIR,
    model_path=MODELS_DIR / "hybrid_recommender.joblib",
    models=None,
    epochs=5,
    device=None,
):
    metadata = ("books", "book_tags", "tags") if models is None else ()
    data = load_training_data(processed_data_dir, raw_data_dir, metadata=metadata)
    if models is None:
        models = build_base_models(**data)
        from src.models.two_tower import build_two_tower

        models["two_tower"] = build_two_tower(**data, device=device).fit(
            load_validation(processed_data_dir), epochs=epochs
        )
    return save_model(HybridRecommender(data["train_df"], models=models), model_path)


if __name__ == "__main__":
    from src.models.hybrid_recommender import train_hybrid_recommender

    train_hybrid_recommender()
