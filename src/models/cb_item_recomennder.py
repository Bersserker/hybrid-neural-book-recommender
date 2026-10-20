"""Коллаборативная фильтрация по книгам из prototype.ipynb."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.models._io import MODELS_DIR, load_training_data, save_model


class cb_item_recomennder:
    def __init__(self, train_df, df_books):
        self.user_history = train_df.groupby("user_id")["book_id"].agg(set).to_dict()
        self.user_ids = np.sort(train_df["user_id"].unique())
        self.book_ids = np.sort(df_books["id"].unique())

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}

        self.book_to_idx = {book_id: idx for idx, book_id in enumerate(self.book_ids)}

        self.rows = train_df["user_id"].map(self.user_to_idx).to_numpy()
        self.cols = train_df["book_id"].map(self.book_to_idx).to_numpy()

        self.interaction_matrix = csr_matrix(
            (train_df["rating"].to_numpy(dtype=np.float32), (self.rows, self.cols)),
            shape=(len(self.user_ids), len(self.book_ids)),
            dtype=np.float32,
        )

        self.item_matrix = self.interaction_matrix.T.tocsr()

        self.item_similarity = cosine_similarity(self.item_matrix, dense_output=False)

        self.item_similarity.setdiag(0)
        self.item_similarity.eliminate_zeros()

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
        if user_id not in self.user_to_idx:
            return self.recommend_from_top(user_id, N)

        user_idx = self.user_to_idx[user_id]
        user_ratings = self.interaction_matrix.getrow(user_idx)
        # Матрица cosine similarity симметрична: обходим только оценки пользователя.
        weighted_scores = (user_ratings @ self.item_similarity).toarray().ravel()

        observed = user_ratings.copy()
        observed.data = np.ones_like(observed.data)
        normalizer = (observed @ self.item_similarity).toarray().ravel()
        scores = np.divide(
            weighted_scores,
            normalizer,
            out=np.zeros_like(weighted_scores),
            where=normalizer > 0,
        )
        scores[normalizer <= 0] = -np.inf
        scores[user_ratings.indices] = -np.inf

        valid_indices = np.where(np.isfinite(scores))[0]
        top_indices = valid_indices[np.argsort(scores[valid_indices])[::-1][:N]]
        return [
            {"book_id": int(self.book_ids[idx]), "score": float(scores[idx])}
            for idx in top_indices
        ]


def train_item_recommender(
    processed_data_dir=PROCESSED_DATA_DIR,
    raw_data_dir=RAW_DATA_DIR,
    model_path=MODELS_DIR / "item_recommender.joblib",
):
    data = load_training_data(processed_data_dir, raw_data_dir, metadata=("books",))
    return save_model(cb_item_recomennder(**data), model_path)


if __name__ == "__main__":
    from src.models.cb_item_recomennder import train_item_recommender

    train_item_recommender()
