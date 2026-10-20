import pickle
from pathlib import Path

import numpy as np

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.load_dataset import laod_dataframes


def clean_ratings(df_ratings, df_books):
    """Отчистка рейтингов"""
    ratings = df_ratings[["user_id", "book_id", "rating"]].copy()

    ratings = ratings.dropna()
    ratings = ratings.groupby(["user_id", "book_id"], as_index=False)["rating"].agg(
        "mean"
    )
    # Оставляем книги, которые присутствуют в каталоге
    ratings = ratings[ratings["book_id"].isin(df_books["id"])]

    return ratings.reset_index(drop=True)


def train_test_val_per_user(
    df, train_size=0.6, val_size=0.2, test_size=0.2, min_ratings=3, random_state=42
):
    """
    Сплит, при котором КАЖДЫЙ пользователь присутствует во всех трёх выборках.
    У каждого пользователя оставляем минимум 1 оценку в train.
    Пользователи с < min_ratings оценками — целиком в train.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    rng = np.random.default_rng(random_state)
    df = df.copy()

    # случайный порядок внутри пользователя
    df["_rand"] = rng.random(len(df))
    df = df.sort_values(["user_id", "_rand"]).reset_index(drop=True)

    df["_rank"] = df.groupby("user_id").cumcount()
    n_per_user = df.groupby("user_id")["_rank"].transform("max") + 1  # Series

    # --- Считаем размеры val/test в numpy, чтобы не ловить pandas-ловушки ---
    n_per_user_np = n_per_user.to_numpy()
    too_few = n_per_user_np < min_ratings

    n_val = np.maximum(np.floor(n_per_user_np * val_size).astype(int), 1)
    n_test = np.maximum(np.floor(n_per_user_np * test_size).astype(int), 1)

    # если у пользователя слишком мало оценок, чтобы дать всем по 1,
    # уменьшаем val/test, чтобы train остался >= 1
    # (страхуемся на случай min_ratings=3 и n=3: 1/1/1 — ок;
    #  но если n=2 и min_ratings=2 — тоже надо не уйти в 0)
    n_val = np.where(
        n_per_user_np - n_val - n_test < 1,
        np.maximum(n_per_user_np - n_test - 1, 0),
        n_val,
    )
    n_test = np.where(
        n_per_user_np - n_val - n_test < 1,
        np.maximum(n_per_user_np - n_val - 1, 0),
        n_test,
    )
    n_train = n_per_user_np - n_val - n_test

    # пользователей с < min_ratings целиком в train
    n_train = np.where(too_few, n_per_user_np, n_train)
    n_val = np.where(too_few, 0, n_val)
    n_test = np.where(too_few, 0, n_test)

    # ---- привязываем размеры обратно к строкам ----
    # n_per_user — Series с индексом df.index, поэтому map по индексу безопасен
    idx = df.index  # noqa: F841 - сохранено соответствие прототипу
    user_vals = df["user_id"].to_numpy()

    # словарь user_id → (n_train, n_val)
    # (проще и надёжнее, чем transform по двум разным колонкам)
    sizes = {}
    seen = {}
    for uid, nt, nv in zip(user_vals, n_train, n_val):
        if uid not in seen:
            seen[uid] = True
            sizes[uid] = (nt, nv)

    df["_n_train"] = df["user_id"].map(lambda u: sizes[u][0])
    df["_n_val"] = df["user_id"].map(lambda u: sizes[u][1])

    is_train = df["_rank"] < df["_n_train"]
    is_val = (df["_rank"] >= df["_n_train"]) & (
        df["_rank"] < df["_n_train"] + df["_n_val"]
    )
    is_test = ~(is_train | is_val)

    cols_drop = ["_rand", "_rank", "_n_train", "_n_val"]
    train_df = df[is_train].drop(columns=cols_drop).reset_index(drop=True)
    val_df = df[is_val].drop(columns=cols_drop).reset_index(drop=True)
    test_df = df[is_test].drop(columns=cols_drop).reset_index(drop=True)

    return train_df, val_df, test_df


def prepare_data(
    raw_data_dir: str | Path = RAW_DATA_DIR,
    processed_data_dir: str | Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """Подготовить данные из прототипа и вернуть пути к сохранённым файлам.

    Очищенные ratings и выборки train/val/test сохраняются в CSV без индекса.
    user_history, val_relevant и test_relevant сохраняются в pickle как
    словари user_id → set(book_id). История строится только по train,
    релевантными считаются оценки >= 4. Повторный запуск перезаписывает файлы.
    """
    dataset_dict = laod_dataframes(Path(raw_data_dir))
    ratings = clean_ratings(dataset_dict["df_ratings"], dataset_dict["df_books"])
    train_df, val_df, test_df = train_test_val_per_user(ratings, 0.6, 0.2, 0.2)

    user_history = train_df.groupby("user_id")["book_id"].agg(set).to_dict()
    val_relevant = (
        val_df[val_df["rating"] >= 4].groupby("user_id")["book_id"].agg(set).to_dict()
    )
    test_relevant = (
        test_df[test_df["rating"] >= 4].groupby("user_id")["book_id"].agg(set).to_dict()
    )

    processed_data_dir = Path(processed_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for name, frame in {
        "ratings": ratings,
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }.items():
        path = processed_data_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        saved_paths[name] = path

    for name, data in {
        "user_history": user_history,
        "val_relevant": val_relevant,
        "test_relevant": test_relevant,
    }.items():
        path = processed_data_dir / f"{name}.pkl"
        with path.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        saved_paths[name] = path

    return saved_paths


if __name__ == "__main__":
    for name, path in prepare_data().items():
        print(f"{name}: {path}")
