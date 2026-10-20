"""Признаки пользователей и книг из prototype.ipynb; статистики только по train."""

import numpy as np
import pandas as pd

GENRE_TAGS = {
    "fantasy": {"fantasy", "high-fantasy", "epic-fantasy", "urban-fantasy"},
    "science_fiction": {
        "science-fiction",
        "sci-fi",
        "scifi",
        "science-fiction-fantasy",
    },
    "romance": {"romance", "contemporary-romance", "historical-romance"},
    "mystery": {"mystery", "mysteries", "crime", "detective"},
    "thriller": {"thriller", "thrillers", "suspense"},
    "horror": {"horror", "paranormal"},
    "historical": {"historical-fiction", "historical", "history"},
    "young_adult": {"young-adult", "ya", "teen", "teen-fiction"},
    "children": {"children", "childrens", "children-s", "middle-grade"},
    "classics": {"classics", "classic", "literary-fiction"},
    "nonfiction": {"non-fiction", "nonfiction", "essays"},
    "biography": {"biography", "biographies", "memoir", "memoirs", "autobiography"},
    "poetry": {"poetry", "poems"},
    "comics": {"comics", "graphic-novels", "graphic-novel", "manga"},
}


class ExtendedFeatureBuilder:
    """Обучающие статистики отделены от преобразования пар (user_id, book_id)."""

    @staticmethod
    def _stats(frame, key, prefix):
        data = frame.assign(
            _square=frame["rating"] ** 2, _positive=frame["rating"] >= 4
        )
        stats = data.groupby(key).agg(
            count=("rating", "size"),
            total=("rating", "sum"),
            square=("_square", "sum"),
            positive=("_positive", "sum"),
        )
        return stats.add_prefix(prefix + "_")

    def fit(self, train, books, book_tags, tags):
        required = train[["user_id", "book_id", "rating"]].copy()
        if required.empty or required.isna().any().any():
            raise ValueError("Нужен непустой train без пропусков")
        if required.duplicated(["user_id", "book_id"]).any():
            raise ValueError("Сначала агрегируйте повторные пары user_id/book_id")
        if not required["book_id"].isin(books["id"]).all():
            raise ValueError("Книги train должны присутствовать в каталоге books.id")
        self.train = required
        self.train_index = required.set_index(["user_id", "book_id"])
        self.global_mean = float(required["rating"].mean())
        self.global_sum = float(required["rating"].sum())
        self.user_stats = self._stats(required, "user_id", "user")
        self.book_stats = self._stats(required, "book_id", "book")
        self.genres = list(GENRE_TAGS)
        catalog = books[["id", "book_id", "title", "authors"]].copy()
        if catalog["id"].duplicated().any():
            raise ValueError("books.id должен быть уникальным")

        # В ratings.book_id хранится books.id; books.book_id — Goodreads ID для тегов.
        tag_to_genre = {
            tag: genre for genre, names in GENRE_TAGS.items() for tag in names
        }
        tagged = book_tags.merge(tags[["tag_id", "tag_name"]], on="tag_id", how="left")
        tagged["genre"] = tagged["tag_name"].str.lower().str.strip().map(tag_to_genre)
        genre_counts = (
            tagged.dropna(subset=["genre"])
            .pivot_table(
                index="goodreads_book_id",
                columns="genre",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(columns=self.genres, fill_value=0)
        )
        counts = genre_counts.reindex(catalog["book_id"], fill_value=0)
        counts.index = catalog["id"].to_numpy()
        self.genre_matrix = (counts > 0).astype(np.float32)
        self.genre_matrix.index.name = "book_id"
        catalog["main_genre"] = np.where(
            counts.sum(axis=1).to_numpy() > 0,
            counts.idxmax(axis=1).to_numpy(),
            "unknown",
        )
        catalog["primary_author"] = (
            catalog["authors"].fillna("").str.split(",").str[0].str.strip()
        )
        self.catalog = (
            catalog.drop(columns=["book_id"])
            .rename(columns={"id": "book_id"})
            .set_index("book_id")
        )

        buckets = required["rating"].round().clip(1, 5).astype(int)
        self.rating_histogram = pd.crosstab(required["book_id"], buckets).reindex(
            columns=range(1, 6), fill_value=0
        )
        positive = required[required["rating"] >= 4]
        positive_genres = self.genre_matrix.reindex(positive["book_id"]).to_numpy()
        self.user_genres = (
            pd.DataFrame(positive_genres, columns=self.genres)
            .groupby(positive["user_id"].to_numpy())
            .sum()
        )
        authors = positive.merge(
            self.catalog[["primary_author"]], on="book_id", how="left"
        )
        self.user_authors = (
            authors[authors["primary_author"] != ""]
            .groupby(["user_id", "primary_author"])
            .size()
        )
        return self

    def transform(self, pairs, leave_one_out=False):
        """rating сохраняется как target и не читается при вычислении признаков.

        leave_one_out=True применяется только к парам train: собственный рейтинг
        исключается из статистик пользователя, книги, автора и жанрового профиля.
        """
        result = pairs.reset_index(drop=True).copy()
        users, books = result["user_id"], result["book_id"]
        if not books.isin(self.catalog.index).all():
            raise ValueError("Для признаков книги нужен её каталог")
        keys = pd.MultiIndex.from_frame(result[["user_id", "book_id"]])
        removed = np.zeros(len(result))
        if leave_one_out:
            source = self.train_index.reindex(keys)["rating"]
            if source.isna().any():
                raise ValueError("leave_one_out допустим только для пар train")
            removed = source.to_numpy()
        subtract = int(leave_one_out)
        positive_removed = (removed >= 4).astype(int)
        prior = np.full(len(result), self.global_mean)
        if leave_one_out:
            prior = (
                (self.global_sum - removed) / (len(self.train) - 1)
                if len(self.train) > 1
                else np.full(len(result), 3.0)
            )

        for prefix, identifiers, stats in [
            ("user", users, self.user_stats),
            ("book", books, self.book_stats),
        ]:
            values = stats.reindex(identifiers).fillna(0)
            n = values[f"{prefix}_count"].to_numpy() - subtract
            total = values[f"{prefix}_total"].to_numpy() - removed
            squares = values[f"{prefix}_square"].to_numpy() - removed**2
            positives = values[f"{prefix}_positive"].to_numpy() - positive_removed
            mean = np.divide(total, n, out=prior.copy(), where=n > 0)
            variance = np.divide(
                squares, n, out=np.zeros(len(result)), where=n > 0
            ) - np.where(n > 0, mean**2, 0)
            result[f"{prefix}_ratings_count"] = n.astype(int)
            result[f"{prefix}_mean_rating"] = mean
            result[f"{prefix}_rating_std"] = np.sqrt(np.maximum(variance, 0))
            result[f"{prefix}_positive_share"] = np.divide(
                positives, n, out=np.zeros(len(result)), where=n > 0
            )
            result[f"{prefix}_log_activity"] = np.log1p(n)
            if prefix == "user":
                user_positive_count = positives

        result["user_activity_group"] = pd.cut(
            result["user_ratings_count"],
            bins=[-1, 4, 49, np.inf],
            labels=["new", "regular", "active"],
        ).astype(str)
        result["book_popularity_share"] = result["book_ratings_count"] / max(
            len(self.train) - subtract, 1
        )
        result["book_smoothed_rating"] = (
            result["book_mean_rating"] * result["book_ratings_count"] + 50 * prior
        ) / (result["book_ratings_count"] + 50)
        histogram = (
            self.rating_histogram.reindex(books)
            .fillna(0)
            .to_numpy(dtype=float, copy=True)
        )
        if leave_one_out:
            histogram[
                np.arange(len(result)), np.rint(removed).clip(1, 5).astype(int) - 1
            ] -= 1
        n_book = result["book_ratings_count"].to_numpy()[:, None]
        probabilities = np.divide(
            histogram, n_book, out=np.zeros_like(histogram), where=n_book > 0
        )
        result["book_rating_entropy"] = -(
            probabilities * np.log(np.maximum(probabilities, 1e-12))
        ).sum(axis=1) / np.log(5)
        result["book_distinct_ratings"] = (histogram > 0).sum(axis=1)

        genre_vectors = self.genre_matrix.reindex(books).to_numpy()
        profiles = self.user_genres.reindex(users).fillna(0).to_numpy(copy=True)
        profiles -= genre_vectors * positive_removed[:, None]
        dot = np.einsum("ij,ij->i", profiles, genre_vectors)
        norms = np.linalg.norm(profiles, axis=1) * np.linalg.norm(genre_vectors, axis=1)
        result["history_genre_cosine"] = np.divide(
            dot, norms, out=np.zeros(len(result)), where=norms > 0
        )
        genre_count = genre_vectors.sum(axis=1)
        affinity_denominator = user_positive_count * genre_count
        result["history_genre_affinity"] = np.divide(
            dot,
            affinity_denominator,
            out=np.zeros(len(result)),
            where=affinity_denominator > 0,
        )
        result["user_history_genre_count"] = (profiles > 0).sum(axis=1)
        result["book_genre_count"] = genre_count.astype(int)
        for idx, genre in enumerate(self.genres):
            result[f"genre_{genre}"] = genre_vectors[:, idx].astype(int)
            result[f"user_genre_{genre}"] = np.divide(
                profiles[:, idx],
                user_positive_count,
                out=np.zeros(len(result)),
                where=user_positive_count > 0,
            )
        result["main_genre"] = books.map(self.catalog["main_genre"])
        primary_authors = books.map(self.catalog["primary_author"])
        author_keys = pd.MultiIndex.from_arrays([users, primary_authors])
        author_counts = (
            self.user_authors.reindex(author_keys).fillna(0).to_numpy(copy=True)
        )
        author_counts -= positive_removed * (primary_authors != "").to_numpy()
        result["history_author_affinity"] = np.divide(
            author_counts,
            user_positive_count,
            out=np.zeros(len(result)),
            where=user_positive_count > 0,
        )
        result["user_book_rating_gap"] = (
            result["book_mean_rating"] - result["user_mean_rating"]
        )
        return result


def build_feature_tables(train_df, df_books, df_book_tags, df_tags):
    """Построить справочники признаков только по переданной train-выборке."""
    feature_builder = ExtendedFeatureBuilder().fit(
        train_df, df_books, df_book_tags, df_tags
    )

    # Справочные таблицы содержат статистики полной train-истории.
    user_snapshot = feature_builder.transform(
        train_df[["user_id", "book_id"]].drop_duplicates("user_id")
    )
    user_feature_columns = [
        column
        for column in user_snapshot.columns
        if column.startswith("user_")
        and column not in {"user_id", "user_book_rating_gap"}
    ]
    user_features = user_snapshot[["user_id", *user_feature_columns]].copy()

    book_snapshot = feature_builder.transform(
        pd.DataFrame(
            {
                "user_id": -1,
                "book_id": df_books["id"].to_numpy(),
            }
        )
    )
    book_feature_columns = [
        column
        for column in book_snapshot.columns
        if (column.startswith("book_") and column != "book_id")
        or column.startswith("genre_")
    ]
    book_features = book_snapshot[
        ["book_id", *book_feature_columns, "main_genre"]
    ].merge(
        feature_builder.catalog[["title", "primary_author"]],
        on="book_id",
        validate="one_to_one",
    )
    return feature_builder, user_features, book_features
