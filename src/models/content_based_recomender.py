"""Контентные рекомендации из prototype.ipynb."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.models._io import MODELS_DIR, load_training_data, save_model


class content_based_recomender:
    def __init__(self, train_df, df_book_tags, df_books, df_tags):
        self.user_history = train_df.groupby("user_id")["book_id"].agg(set).to_dict()

        # 1. Объединяем теги с их названиями
        self.book_tags = df_book_tags.merge(df_tags, on="tag_id", how="left")

        # 2. Выбираем 10 наиболее значимых тегов для каждой книги
        self.book_tags = (
            self.book_tags.sort_values("count", ascending=False)
            .groupby("goodreads_book_id")
            .head(10)
            .copy()
        )

        # 3. Подготавливаем названия тегов
        self.book_tags["tag_name"] = (
            self.book_tags["tag_name"].fillna("").str.replace("-", " ", regex=False)
        )

        # 4. Объединяем теги в одну строку для каждой книги
        self.tags_aggregated = (
            self.book_tags.groupby("goodreads_book_id")["tag_name"]
            .agg(" ".join)
            .reset_index()
        )

        # 5. Создаём таблицу с текстовым описанием книг
        self.books_content = df_books[
            ["id", "book_id", "title", "original_title"]
        ].copy()

        # 6. Объединяем книги с тегами по разным названиям ключа
        self.books_content = self.books_content.merge(
            self.tags_aggregated,
            left_on="book_id",
            right_on="goodreads_book_id",
            how="left",
        )

        # 7. Формируем текст для TF-IDF
        self.books_content["content"] = (
            self.books_content["original_title"]
            .fillna(self.books_content["title"])
            .fillna("")
            + " "
            + self.books_content["tag_name"].fillna("")
        )

        self.tfidf = TfidfVectorizer(
            stop_words="english", max_features=20000, ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.tfidf.fit_transform(self.books_content["content"])

        # TfidfVectorizer уже нормализует строки по L2.
        self.tfidf_transpose = self.tfidf_matrix.T.tocsr()

        self.book_id_to_idx = {
            int(book_id): idx for idx, book_id in enumerate(self.books_content["id"])
        }

        self.positive_history = (
            train_df[train_df["rating"] >= 4]
            .groupby("user_id")["book_id"]
            .agg(list)
            .to_dict()
        )

        self.global_mean = train_df["rating"].mean()
        self.m = 50

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

    def recommend_from_top(self, user_id, N=10):
        seen = self.user_history.get(user_id, set())

        candidates = self.book_popularity[
            ~self.book_popularity["book_id"].isin(seen)
        ].head(N)

        return [
            {"book_id": int(row.book_id), "score": float(row.score)}
            for row in candidates.itertuples()
        ]

    def recommend(self, user_id, N=10):
        if N <= 0:
            return []
        liked_books = self.positive_history.get(user_id, [])

        indices = [
            self.book_id_to_idx[book_id]
            for book_id in liked_books
            if book_id in self.book_id_to_idx
        ]

        if not indices:
            return self.recommend_from_top(user_id, N)

        user_profile = np.asarray(self.tfidf_matrix[indices].mean(axis=0))

        user_profile = normalize(user_profile)

        scores = (csr_matrix(user_profile) @ self.tfidf_transpose).toarray().ravel()

        seen = self.user_history.get(user_id, set())

        for book_id in seen:
            idx = self.book_id_to_idx.get(book_id)
            if idx is not None:
                scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:N]

        return [
            {
                "book_id": int(self.books_content.iloc[idx]["id"]),
                "score": float(scores[idx]),
            }
            for idx in top_indices
            if np.isfinite(scores[idx])
        ]


def train_content_based_recommender(
    processed_data_dir=PROCESSED_DATA_DIR,
    raw_data_dir=RAW_DATA_DIR,
    model_path=MODELS_DIR / "content_based_recommender.joblib",
):
    data = load_training_data(
        processed_data_dir, raw_data_dir, metadata=("books", "book_tags", "tags")
    )
    return save_model(content_based_recomender(**data), model_path)


if __name__ == "__main__":
    from src.models.content_based_recomender import train_content_based_recommender

    train_content_based_recommender()
