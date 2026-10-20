"""Популярностная модель с байесовским сглаживанием рейтингов из прототипа."""

from pathlib import Path

import joblib
import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_DIR


class recommend_popular_recomemder:
    def __init__(
        self,
        train_df,
        m=50,
    ):
        if train_df.empty:
            raise ValueError("Обучающая выборка пуста")
        if not 0 <= m < float("inf"):
            raise ValueError("m должен быть конечным неотрицательным числом")
        self.user_history = train_df.groupby("user_id")["book_id"].agg(set).to_dict()
        self.global_mean = train_df["rating"].mean()
        self.m = m

        self.book_popularity = (
            train_df.groupby("book_id")
            .agg(mean_rating=("rating", "mean"), ratings_count=("rating", "count"))
            .reset_index()
        )

        self.book_popularity["score"] = (
            self.book_popularity["ratings_count"]
            / (self.book_popularity["ratings_count"] + self.m)
            * self.book_popularity["mean_rating"]
            + self.m
            / (self.book_popularity["ratings_count"] + self.m)
            * self.global_mean
        )
        self.book_popularity = self.book_popularity.sort_values(
            "score", ascending=False, kind="stable"
        )

    def recommend(self, user_id, N=10):
        if N <= 0:
            return []
        seen = self.user_history.get(user_id, set())

        candidates = self.book_popularity[
            ~self.book_popularity["book_id"].isin(seen)
        ].head(N)

        return [
            {"book_id": int(row.book_id), "score": float(row.score)}
            for row in candidates.itertuples()
        ]


def train_popular_recommender(
    processed_data_dir: str | Path = PROCESSED_DATA_DIR,
    model_path: str | Path = PROJECT_DIR / "artifacts" / "popular_recommender.joblib",
    m: float = 50,
) -> recommend_popular_recomemder:
    """Обучить модель на train.csv, сохранить через joblib и вернуть её."""
    train_df = pd.read_csv(Path(processed_data_dir) / "train.csv")
    model = recommend_popular_recomemder(train_df, m=m)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


if __name__ == "__main__":
    # Импорт по имени модуля позволяет загрузить joblib в другом процессе.
    from src.models.recommend_popular_recomemder import train_popular_recommender

    train_popular_recommender()
    print("Модель сохранена: artifacts/popular_recommender.joblib")
