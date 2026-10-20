"""Загрузка входных данных и сохранение обученных моделей."""

import pickle
from pathlib import Path

import joblib
import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROJECT_DIR, RAW_DATA_DIR

MODELS_DIR = PROJECT_DIR / "artifacts"


def load_training_data(
    processed_data_dir=PROCESSED_DATA_DIR, raw_data_dir=RAW_DATA_DIR, metadata=()
):
    """Читать только train и запрошенные общедоступные таблицы каталога."""
    data = {"train_df": pd.read_csv(Path(processed_data_dir) / "train.csv")}
    for name in metadata:
        data[f"df_{name}"] = pd.read_csv(Path(raw_data_dir) / f"{name}.csv")
    return data


def load_validation(processed_data_dir=PROCESSED_DATA_DIR):
    with (Path(processed_data_dir) / "val_relevant.pkl").open("rb") as file:
        return pickle.load(file)


def save_model(model, model_path):
    """Сохранить модель; файл можно загрузить через joblib.load."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model
