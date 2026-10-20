"""Матричная факторизация Surprise SVD из prototype.ipynb."""

import numpy as np
from surprise import SVD, Dataset, Reader

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.models._io import MODELS_DIR, load_training_data, save_model


class svd_recommender:
    def __init__(
        self,
        train_df,
        df_books,
        n_epochs=50,
        random_state=42,
        *,
        n_factors=50,
        lr_all=0.005,
        reg_all=0.02,
    ):
        if train_df.empty:
            raise ValueError("Обучающая выборка пуста")
        self.reader = Reader(rating_scale=(1, 5))

        dataset = Dataset.load_from_df(
            train_df[["user_id", "book_id", "rating"]], self.reader
        )

        self.trainset = dataset.build_full_trainset()

        self.svd_model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state,
        )

        self.svd_model.fit(self.trainset)

        self.user_ids = np.sort(train_df["user_id"].unique())
        self.book_ids = np.sort(df_books["id"].unique())

        self.user_history = train_df.groupby("user_id")["book_id"].agg(set).to_dict()

        # Выровненные по каталогу факторы; неизвестные книги имеют нулевые факторы.
        self.catalog_factors = np.zeros((len(self.book_ids), self.svd_model.n_factors))
        self.catalog_bias = np.zeros(len(self.book_ids))
        book_to_idx = {int(b): i for i, b in enumerate(self.book_ids)}
        for inner in self.trainset.all_items():
            idx = book_to_idx[int(self.trainset.to_raw_iid(inner))]
            self.catalog_factors[idx] = self.svd_model.qi[inner]
            self.catalog_bias[idx] = self.svd_model.bi[inner]
        self.user_to_inner = {
            self.trainset.to_raw_uid(i): i for i in self.trainset.all_users()
        }

    def recommend(self, user_id, N=10):
        if N <= 0:
            return []
        scores = self.trainset.global_mean + self.catalog_bias
        inner = self.user_to_inner.get(user_id)
        if inner is not None:
            # Та же формула Surprise SVD, сразу для всего каталога.
            scores = (
                scores
                + self.svd_model.bu[inner]
                + np.einsum("ij,j->i", self.catalog_factors, self.svd_model.pu[inner])
            )
        scores = np.clip(scores, *self.trainset.rating_scale)
        seen = self.user_history.get(user_id, set())
        scores[np.isin(self.book_ids, list(seen))] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        top = valid[np.argsort(-scores[valid], kind="stable")[:N]]
        return [
            {"book_id": int(self.book_ids[i]), "score": float(scores[i])} for i in top
        ]


def train_svd_recommender(
    processed_data_dir=PROCESSED_DATA_DIR,
    raw_data_dir=RAW_DATA_DIR,
    model_path=MODELS_DIR / "svd_recommender.joblib",
    **kwargs,
):
    data = load_training_data(processed_data_dir, raw_data_dir, metadata=("books",))
    return save_model(svd_recommender(**data, **kwargs), model_path)


if __name__ == "__main__":
    from src.models.svd_recommender import train_svd_recommender

    train_svd_recommender()
