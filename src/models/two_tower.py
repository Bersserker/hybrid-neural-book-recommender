"""Two Tower: ID-эмбеддинги, BPR loss и выбор лучшей эпохи по val."""

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.features import build_feature_tables
from src.metrics.evaluation import evaluate_model
from src.models._io import MODELS_DIR, load_training_data, load_validation, save_model


class TwoTower(nn.Module):
    """Каждая башня получает ID-эмбеддинг и собственные числовые признаки."""

    def __init__(
        self,
        n_users,
        n_books,
        embedding_dim=64,
        output_dim=32,
        user_feature_dim=0,
        book_feature_dim=0,
    ):
        super().__init__()
        self.user_feature_dim = user_feature_dim
        self.book_feature_dim = book_feature_dim
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.book_embedding = nn.Embedding(n_books, embedding_dim)

        def tower(feature_dim):
            return nn.Sequential(
                nn.Linear(embedding_dim + feature_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, output_dim),
            )

        self.user_tower = tower(user_feature_dim)
        self.book_tower = tower(book_feature_dim)

    def encode_user(self, users, features=None):
        embedded = self.user_embedding(users)
        if self.user_feature_dim:
            if features is None:
                raise ValueError("Нужны числовые признаки пользователей")
            embedded = torch.cat([embedded, features], dim=-1)
        return F.normalize(self.user_tower(embedded), dim=-1)

    def encode_book(self, books, features=None):
        embedded = self.book_embedding(books)
        if self.book_feature_dim:
            if features is None:
                raise ValueError("Нужны числовые признаки книг")
            embedded = torch.cat([embedded, features], dim=-1)
        return F.normalize(self.book_tower(embedded), dim=-1)

    def forward(self, users, books, user_features=None, book_features=None):
        return (
            self.encode_user(users, user_features)
            * self.encode_book(books, book_features)
        ).sum(dim=-1)


class TwoTowerRecommender:
    def __init__(
        self,
        train_df,
        df_books,
        train_features=None,
        user_features=None,
        book_features=None,
        use_features=True,
        embedding_dim=64,
        output_dim=32,
        random_state=42,
        device=None,
    ):
        if train_df.empty or df_books.empty:
            raise ValueError("Нужны непустые train и каталог")
        if not train_df["book_id"].isin(df_books["id"]).all():
            raise ValueError("Книги train должны присутствовать в books.id")
        if df_books["id"].duplicated().any():
            raise ValueError("books.id должен быть уникальным")
        torch.manual_seed(random_state)
        self.random_state = random_state
        self.use_features = use_features
        self.rng = np.random.default_rng(random_state)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.user_ids = np.sort(train_df["user_id"].unique())
        self.book_ids = np.sort(df_books["id"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.book_to_idx = {b: i for i, b in enumerate(self.book_ids)}
        self.user_history = train_df.groupby("user_id")["book_id"].agg(set).to_dict()
        self.known_books = np.isin(self.book_ids, train_df["book_id"].unique())
        self.popular = train_df.groupby("book_id").size().sort_values(ascending=False)
        users = train_df["user_id"].map(self.user_to_idx).to_numpy(dtype=np.int64)
        books = train_df["book_id"].map(self.book_to_idx).to_numpy(dtype=np.int64)
        self.seen_keys = np.unique(users * len(self.book_ids) + books)
        can_sample = (
            train_df["user_id"]
            .map(
                {
                    u: len(seen) < len(self.book_ids)
                    for u, seen in self.user_history.items()
                }
            )
            .to_numpy()
        )
        positive = (train_df["rating"].to_numpy() >= 4) & can_sample
        self.pairs = np.column_stack((users[positive], books[positive]))
        self.positive_ratings = train_df["rating"].to_numpy()[positive]
        self.train_size = len(train_df)
        self.train_rating_sum = float(train_df["rating"].sum())
        if not len(self.pairs):
            raise ValueError("Нужны положительные train-пары и непрочитанные книги")

        self.user_feature_columns = []
        self.book_feature_columns = []
        self.user_scaler = self.book_scaler = None
        full_user = np.empty((len(self.user_ids), 0), dtype=np.float32)
        full_book = np.empty((len(self.book_ids), 0), dtype=np.float32)
        pair_user = np.empty((len(self.pairs), 0), dtype=np.float32)
        pair_book = np.empty((len(self.pairs), 0), dtype=np.float32)
        if use_features:
            if any(
                table is None
                for table in [train_features, user_features, book_features]
            ):
                raise ValueError(
                    "Передайте train_features (leave-one-out), user_features и book_features"
                )
            if (
                user_features["user_id"].duplicated().any()
                or book_features["book_id"].duplicated().any()
            ):
                raise ValueError("Таблицы признаков должны содержать уникальные ID")
            if train_features.duplicated(["user_id", "book_id"]).any():
                raise ValueError("В train_features должны быть уникальные пары")
            self.user_feature_columns = [
                c
                for c in user_features.select_dtypes(include="number").columns
                if c.startswith("user_") and c != "user_id"
            ]
            self.book_feature_columns = [
                c
                for c in book_features.select_dtypes(include="number").columns
                if (c.startswith("book_") and c != "book_id") or c.startswith("genre_")
            ]
            if not self.user_feature_columns or not self.book_feature_columns:
                raise ValueError("Нужны числовые признаки обеих башен")
            pair_keys = pd.MultiIndex.from_frame(train_df[["user_id", "book_id"]])
            aligned = train_features.set_index(["user_id", "book_id"]).reindex(
                pair_keys
            )
            if len(train_features) != len(train_df) or not np.array_equal(
                aligned["rating"].to_numpy(), train_df["rating"].to_numpy()
            ):
                raise ValueError(
                    "train_features должны соответствовать обучающей выборке"
                )
            # Эти счётчики позволяют обнаружить случайную передачу full-train вместо LOO.
            expected_user_counts = (
                train_df["user_id"].map(train_df.groupby("user_id").size()).to_numpy()
                - 1
            )
            expected_book_counts = (
                train_df["book_id"].map(train_df.groupby("book_id").size()).to_numpy()
                - 1
            )
            if not (
                np.array_equal(aligned["user_ratings_count"], expected_user_counts)
                and np.array_equal(aligned["book_ratings_count"], expected_book_counts)
            ):
                raise ValueError("Для обучения нужны признаки leave-one-out")
            full_user = user_features.set_index("user_id").reindex(self.user_ids)[
                self.user_feature_columns
            ]
            full_book = book_features.set_index("book_id").reindex(self.book_ids)[
                self.book_feature_columns
            ]
            loo_user = aligned[self.user_feature_columns]
            loo_book = aligned[self.book_feature_columns]
            for frame in [full_user, full_book, loo_user, loo_book]:
                if not np.isfinite(frame.to_numpy()).all():
                    raise ValueError("Все ID должны иметь конечные числовые признаки")
            # Fit только на обучающих строках. Тот же scaler применяется при выдаче.
            self.raw_book_features = full_book.to_numpy(copy=True)
            self.user_scaler = StandardScaler().fit(loo_user)
            self.book_scaler = StandardScaler().fit(loo_book)
            full_user = self.user_scaler.transform(full_user).astype(np.float32)
            full_book = self.book_scaler.transform(full_book).astype(np.float32)
            pair_user = self.user_scaler.transform(
                loo_user.iloc[np.flatnonzero(positive)]
            ).astype(np.float32)
            pair_book = self.book_scaler.transform(
                loo_book.iloc[np.flatnonzero(positive)]
            ).astype(np.float32)

        # При выдаче используются полные train-профили; при обучении — LOO-профили.
        self.user_feature_tensor = torch.tensor(
            full_user, dtype=torch.float32, device=self.device
        )
        self.book_feature_tensor = torch.tensor(
            full_book, dtype=torch.float32, device=self.device
        )
        self.pair_user_features = torch.tensor(pair_user, dtype=torch.float32)
        self.pair_book_features = torch.tensor(pair_book, dtype=torch.float32)
        self.network = TwoTower(
            len(self.user_ids),
            len(self.book_ids),
            embedding_dim,
            output_dim,
            len(self.user_feature_columns),
            len(self.book_feature_columns),
        ).to(self.device)
        self._cache_vectors()

    @torch.no_grad()
    def _cache_vectors(self, batch_size=4096):
        self.network.eval()

        def encode_all(size, encoder, features):
            vectors = []
            for start in range(0, size, batch_size):
                stop = min(start + batch_size, size)
                indices = torch.arange(start, stop, device=self.device)
                vectors.append(encoder(indices, features[start:stop]).cpu().numpy())
            return np.concatenate(vectors)

        self.user_vectors = encode_all(
            len(self.user_ids), self.network.encode_user, self.user_feature_tensor
        )
        self.book_vectors = encode_all(
            len(self.book_ids), self.network.encode_book, self.book_feature_tensor
        )

    def _sample_negatives(self):
        # Не считаем отрицательными и книги с низкими train-оценками пользователя.
        negatives = self.rng.integers(len(self.book_ids), size=len(self.pairs))
        invalid = np.isin(
            self.pairs[:, 0] * len(self.book_ids) + negatives, self.seen_keys
        )
        while invalid.any():
            negatives[invalid] = self.rng.integers(
                len(self.book_ids), size=invalid.sum()
            )
            invalid = np.isin(
                self.pairs[:, 0] * len(self.book_ids) + negatives, self.seen_keys
            )
        return negatives

    def _negative_book_features(self, negatives):
        """И у отрицательной книги исключаем target из глобального prior."""
        if not self.use_features:
            return torch.empty((len(negatives), 0), dtype=torch.float32)
        raw = pd.DataFrame(
            self.raw_book_features[negatives].copy(), columns=self.book_feature_columns
        )
        n = raw["book_ratings_count"].to_numpy()
        old_prior = self.train_rating_sum / self.train_size
        prior = (
            (self.train_rating_sum - self.positive_ratings) / (self.train_size - 1)
            if self.train_size > 1
            else np.full(len(negatives), 3.0)
        )
        raw["book_smoothed_rating"] += 50 * (prior - old_prior) / (n + 50)
        raw.loc[n == 0, "book_mean_rating"] = prior[n == 0]
        raw["book_popularity_share"] = n / max(self.train_size - 1, 1)
        return torch.tensor(self.book_scaler.transform(raw), dtype=torch.float32)

    def fit(self, val_relevant, epochs=5, batch_size=1024, lr=1e-3, k=10):
        if not val_relevant or epochs < 1 or batch_size < 1 or lr <= 0 or k < 1:
            raise ValueError(
                "Нужны непустая val и положительные epochs, batch_size, lr, k"
            )
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=1e-5
        )
        generator = torch.Generator().manual_seed(self.random_state)
        best_score, best_state, history = -np.inf, None, []
        for epoch in range(1, epochs + 1):
            negatives = self._sample_negatives()
            negative_features = self._negative_book_features(negatives)
            loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(self.pairs[:, 0].copy()),
                    torch.from_numpy(self.pairs[:, 1].copy()),
                    torch.from_numpy(negatives),
                    self.pair_user_features,
                    self.pair_book_features,
                    negative_features,
                ),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )
            self.network.train()
            total_loss = 0.0
            for batch in loader:
                (
                    users,
                    positive,
                    negative,
                    user_features_batch,
                    positive_features,
                    negative_features,
                ) = [x.to(self.device) for x in batch]
                positive_score = self.network(
                    users, positive, user_features_batch, positive_features
                )
                negative_score = self.network(
                    users, negative, user_features_batch, negative_features
                )
                loss = F.softplus((negative_score - positive_score) / 0.1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(users)
            self._cache_vectors()
            score = float(
                evaluate_model(self.recommend, val_relevant, k=k)["ndcg"].mean()
            )
            if not np.isfinite(score):
                raise ValueError("Validation nDCG должен быть конечным")
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / len(self.pairs),
                    f"val_nDCG@{k}": score,
                }
            )
            if score > best_score:
                best_score, self.best_epoch = score, epoch
                best_state = copy.deepcopy(self.network.state_dict())
            print(
                f"Epoch {epoch}: loss={history[-1]['loss']:.4f}, val nDCG@{k}={score:.4f}"
            )
        self.network.load_state_dict(best_state)
        self._cache_vectors()
        self.history = pd.DataFrame(history)
        return self

    def recommend(self, user_id, N=10):
        if N <= 0:
            return []
        seen = self.user_history.get(user_id, set())
        if user_id not in self.user_to_idx:
            return [
                {"book_id": int(b), "score": float(count)}
                for b, count in self.popular.items()
                if b not in seen
            ][:N]
        scores = self.book_vectors @ self.user_vectors[self.user_to_idx[user_id]]
        scores[~self.known_books | np.isin(self.book_ids, list(seen))] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        top = valid[np.lexsort((self.book_ids[valid], -scores[valid]))[:N]]
        return [
            {"book_id": int(self.book_ids[i]), "score": float(scores[i])} for i in top
        ]


def train_two_tower_recommender(
    processed_data_dir=PROCESSED_DATA_DIR,
    raw_data_dir=RAW_DATA_DIR,
    model_path=MODELS_DIR / "two_tower_recommender.joblib",
    epochs=5,
    batch_size=1024,
    lr=1e-3,
    k=10,
    **kwargs,
):
    data = load_training_data(
        processed_data_dir, raw_data_dir, metadata=("books", "book_tags", "tags")
    )
    relevant = load_validation(processed_data_dir)
    model = build_two_tower(**data, **kwargs)
    model.fit(relevant, epochs=epochs, batch_size=batch_size, lr=lr, k=k)
    # CPU-артефакт загружается также на машине без CUDA.
    model.device = torch.device("cpu")
    model.network.to(model.device)
    model._cache_vectors()
    return save_model(model, model_path)


def build_two_tower(train_df, df_books, df_book_tags, df_tags, **kwargs):
    """Подготовить train-only и leave-one-out признаки для нейросети."""
    builder, user_features, book_features = build_feature_tables(
        train_df, df_books, df_book_tags, df_tags
    )
    return TwoTowerRecommender(
        train_df,
        df_books,
        train_features=builder.transform(train_df, leave_one_out=True),
        user_features=user_features,
        book_features=book_features,
        **kwargs,
    )


if __name__ == "__main__":
    from src.models.two_tower import train_two_tower_recommender

    train_two_tower_recommender()
