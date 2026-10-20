"""Обучение и сохранение моделей из prototype.ipynb одним запуском."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.models._io import MODELS_DIR
from src.models.recommend_popular_recomemder import train_popular_recommender

MODEL_FILES = {
    "pop": "popular_recommender.joblib",
    "content": "content_based_recommender.joblib",
    "item": "item_recommender.joblib",
    "svd": "svd_recommender.joblib",
    "two_tower": "two_tower_recommender.joblib",
    "hybrid": "hybrid_recommender.joblib",
    "adaptive": "adaptive_hybrid_recommender.joblib",
    "balanced": "balanced_recommender.joblib",
}


def train_models(
    model="all",
    processed_data_dir=PROCESSED_DATA_DIR,
    raw_data_dir=RAW_DATA_DIR,
    models_dir=MODELS_DIR,
    epochs=5,
    n_trials=30,
    device=None,
):
    """Вернуть пути к артефактам; all обучает базовые модели только один раз."""
    if model not in {*MODEL_FILES, "all"}:
        raise ValueError(f"Неизвестная модель: {model}")
    models_dir = Path(models_dir)
    paths = {}
    base_models = {}
    selected = list(MODEL_FILES) if model == "all" else [model]
    for name in selected:
        path = models_dir / MODEL_FILES[name]
        print(f"Обучение {name}...", flush=True)
        started_at = datetime.now(UTC).isoformat()
        started = perf_counter()
        if name == "pop":
            base_models[name] = train_popular_recommender(processed_data_dir, path)
        elif name == "content":
            from src.models.content_based_recomender import (
                train_content_based_recommender,
            )

            base_models[name] = train_content_based_recommender(
                processed_data_dir, raw_data_dir, path
            )
        elif name == "item":
            from src.models.cb_item_recomennder import train_item_recommender

            base_models[name] = train_item_recommender(
                processed_data_dir, raw_data_dir, path
            )
        elif name == "svd":
            from src.models.svd_recommender import train_svd_recommender

            base_models[name] = train_svd_recommender(
                processed_data_dir, raw_data_dir, path
            )
        elif name == "hybrid":
            from src.models.hybrid_recommender import train_hybrid_recommender

            hybrid = train_hybrid_recommender(
                processed_data_dir,
                raw_data_dir,
                path,
                models=base_models or None,
                epochs=epochs,
                device=device,
            )
            base_models = hybrid.models
        elif name == "adaptive":
            from src.models.adaptive_hybrid_recommender import (
                train_adaptive_hybrid_recommender,
            )

            adaptive = train_adaptive_hybrid_recommender(
                processed_data_dir,
                raw_data_dir,
                path,
                models=base_models or None,
                n_trials=n_trials,
                epochs=epochs,
                device=device,
            )
            adaptive.search_results.to_csv(
                models_dir / "adaptive_weight_search.csv", index=False
            )
        elif name == "two_tower":
            from src.models.two_tower import train_two_tower_recommender

            tower = train_two_tower_recommender(
                processed_data_dir, raw_data_dir, path, epochs=epochs, device=device
            )
            tower.history.to_csv(models_dir / "two_tower_history.csv", index=False)
            base_models[name] = tower
        elif name == "balanced":
            from src.models.balanced_recommender import train_balanced_recommender

            train_balanced_recommender(
                processed_data_dir,
                raw_data_dir,
                path,
                model=adaptive if model == "all" else None,
                n_trials=n_trials,
                epochs=epochs,
                device=device,
            )
        metadata = {
            "model": name,
            "started_at": started_at,
            "duration_seconds": perf_counter() - started,
            "model_size_bytes": path.stat().st_size,
            "processed_data_dir": str(Path(processed_data_dir).resolve()),
            "raw_data_dir": str(Path(raw_data_dir).resolve()),
            "epochs": epochs if name == "two_tower" else None,
            "n_trials": n_trials if name == "adaptive" else None,
        }
        path.with_suffix(".training.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        paths[name] = path
        print(f"Сохранено: {path}", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODEL_FILES], default="all")
    parser.add_argument("--processed-data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Число trials для каждой группы адаптивного гибрида; 0 — исходные веса",
    )
    parser.add_argument(
        "--device", default=None, help="cpu, cuda; по умолчанию автоматически"
    )
    args = parser.parse_args()
    train_models(**vars(args))


if __name__ == "__main__":
    main()
