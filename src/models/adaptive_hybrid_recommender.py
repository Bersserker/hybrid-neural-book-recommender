"""Адаптивный гибрид и подбор весов только по validation."""

import numpy as np
import optuna
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.metrics.metrics import ndcg_at_k
from src.models._io import MODELS_DIR, load_training_data, load_validation, save_model
from src.models.hybrid_recommender import HybridRecommender, build_base_models

ADAPTIVE_WEIGHTS = {
    "new": {
        "pop": 0.65,
        "content": 0.25,
        "item": 0.05,
        "svd": 0.05,
    },
    "regular": {
        "pop": 0.15,
        "content": 0.35,
        "item": 0.25,
        "svd": 0.25,
    },
    "active": {
        "pop": 0.05,
        "content": 0.15,
        "item": 0.35,
        "svd": 0.45,
    },
}


class AdaptiveHybridRecommender(HybridRecommender):
    def __init__(self, train_df, models, adaptive_weights=None):
        # Инициализируем базовые модели и историю пользователей
        super().__init__(train_df, models=models)

        self.user_activity = train_df.groupby("user_id").size().to_dict()

        if adaptive_weights is None:
            # 20% изначально отдаём нейросети; остальные веса сохраняют прежние пропорции.
            source_weights = {
                group: {
                    **{name: 0.8 * weight for name, weight in weights.items()},
                    "two_tower": 0.2,
                }
                for group, weights in ADAPTIVE_WEIGHTS.items()
            }
            self.adaptive_weights = {}
            for group, weights in source_weights.items():
                total = sum(weights[name] for name in self.models)
                self.adaptive_weights[group] = {
                    name: weights[name] / total for name in self.models
                }
        else:
            self.adaptive_weights = {
                group: dict(weights) for group, weights in adaptive_weights.items()
            }

    def get_user_type(self, user_id):
        n_ratings = self.user_activity.get(user_id, 0)

        if n_ratings < 5:
            return "new"
        elif n_ratings < 50:
            return "regular"
        else:
            return "active"

    def recommend(self, user_id, N=10, weights=None, candidates_k=100):
        # Явно переданные веса имеют приоритет
        if weights is None:
            user_type = self.get_user_type(user_id)
            weights = self.adaptive_weights[user_type]

        # Объединение рекомендаций и проверку весов выполняет родитель
        return super().recommend(
            user_id=user_id,
            N=N,
            weights=weights,
            candidates_k=candidates_k,
        )


def optimize_adaptive_weights(
    model, val_relevant, n_trials=30, k=10, candidates_k=100, random_state=42
):
    if n_trials < 1:
        raise ValueError("n_trials должен быть положительным")
    best_weights = {group: dict(w) for group, w in model.adaptive_weights.items()}
    results = []
    names = list(model.models)

    for user_type in ("new", "regular", "active"):
        relevant = {
            u: actual
            for u, actual in val_relevant.items()
            if model.get_user_type(u) == user_type
        }
        if not relevant:
            continue  # Для пустой группы сохраняем исходные веса.

        # Рекомендации базовых моделей считаем один раз, переиспользуем в trials.
        cache = {}
        for user_id in relevant:
            normalized = [
                model.normalize_scores(base.recommend(user_id, N=candidates_k))
                for base in model.models.values()
            ]
            ids = sorted(
                set().union(*(scores.keys() for scores in normalized))
                - model.user_history.get(user_id, set())
            )
            matrix = np.array(
                [
                    [scores.get(book_id, 0.0) for scores in normalized]
                    for book_id in ids
                ],
                dtype=float,
            ).reshape(-1, len(names))
            cache[user_id] = (np.asarray(ids, dtype=int), matrix)

        def objective(trial, relevant=relevant, cache=cache, user_type=user_type):
            raw = np.array([trial.suggest_float(name, 1e-6, 1.0) for name in names])
            weights = raw / raw.sum()
            scores = []
            for user_id, actual in relevant.items():
                ids, matrix = cache[user_id]
                order = np.lexsort((ids, -(matrix @ weights)))[:k]
                scores.append(ndcg_at_k(actual, ids[order], k))
            score = float(np.mean(scores))
            trial.set_user_attr("weights", dict(zip(names, weights.tolist())))
            results.append(
                {
                    "user_type": user_type,
                    "trial": trial.number,
                    **trial.user_attrs["weights"],
                    f"nDCG@{k}": score,
                }
            )
            return score

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        # Исходные веса тоже участвуют: результат не хуже исходного на val.
        study.enqueue_trial(
            {name: max(best_weights[user_type][name], 1e-6) for name in names}
        )
        study.optimize(objective, n_trials=n_trials)
        best_weights[user_type] = study.best_trial.user_attrs["weights"]

    return best_weights, pd.DataFrame(results)


def train_adaptive_hybrid_recommender(
    processed_data_dir=PROCESSED_DATA_DIR,
    raw_data_dir=RAW_DATA_DIR,
    model_path=MODELS_DIR / "adaptive_hybrid_recommender.joblib",
    models=None,
    n_trials=30,
    k=10,
    candidates_k=100,
    epochs=5,
    device=None,
):
    metadata = ("books", "book_tags", "tags") if models is None else ()
    data = load_training_data(processed_data_dir, raw_data_dir, metadata=metadata)
    if models is None:
        from src.models.two_tower import build_two_tower

        models = build_base_models(**data)
        models["two_tower"] = build_two_tower(**data, device=device).fit(
            load_validation(processed_data_dir), epochs=epochs, k=k
        )
    model = AdaptiveHybridRecommender(data["train_df"], models=models)
    if n_trials < 0:
        raise ValueError("n_trials должен быть неотрицательным")
    if n_trials:
        relevant = load_validation(processed_data_dir)
        if not relevant:
            raise ValueError("Для подбора весов нужна непустая val")
        model.adaptive_weights, model.search_results = optimize_adaptive_weights(
            model, relevant, n_trials=n_trials, k=k, candidates_k=candidates_k
        )
    else:
        model.search_results = pd.DataFrame(
            columns=["user_type", "trial", *model.models, f"nDCG@{k}"]
        )
    return save_model(model, model_path)


if __name__ == "__main__":
    from src.models.adaptive_hybrid_recommender import train_adaptive_hybrid_recommender

    train_adaptive_hybrid_recommender()
