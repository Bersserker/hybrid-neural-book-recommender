"""Оценка качества по активности пользователей в train."""

import numpy as np
import pandas as pd

SEGMENT_ORDER = ["new", "regular", "active"]


def make_user_segments(train_df, relevant):
    """Сегмент зависит только от переданной обучающей истории."""
    # Границы совпадают с AdaptiveHybridRecommender.get_user_type.
    train_activity = train_df.groupby("user_id").size()
    all_user_ids = train_activity.index.union(pd.Index(list(relevant)))
    user_segments = pd.DataFrame({"user_id": all_user_ids})
    user_segments["train_ratings"] = (
        user_segments["user_id"].map(train_activity).fillna(0).astype(int)
    )
    user_segments["Segment"] = pd.cut(
        user_segments["train_ratings"], bins=[-1, 4, 49, np.inf], labels=SEGMENT_ORDER
    )
    user_segments["in_train"] = user_segments["train_ratings"] > 0
    user_segments["evaluated_on_test"] = user_segments["user_id"].isin(relevant)
    user_segments["cold_start_on_test"] = (
        user_segments["evaluated_on_test"] & ~user_segments["in_train"]
    )

    segment_population = user_segments.groupby("Segment", observed=False).agg(
        train_users=("in_train", "sum"),
        evaluated_test_users=("evaluated_on_test", "sum"),
        cold_start_test_users=("cold_start_on_test", "sum"),
    )
    segment_population["test_user_share"] = segment_population[
        "evaluated_test_users"
    ] / max(len(relevant), 1)
    return user_segments, segment_population


def summarize_segments(metrics_by_model, segments, k=10):
    """Усреднить уже рассчитанные per-user метрики, не вызывая модели повторно."""
    segment_map = segments.set_index("user_id")["Segment"]
    rows = []
    for model_name, metrics in metrics_by_model.items():
        metrics = metrics.copy()
        metrics["Segment"] = metrics["user_id"].map(segment_map)
        if metrics["Segment"].isna().any():
            raise ValueError("Не для всех пользователей определён сегмент")
        for segment in SEGMENT_ORDER:
            group = metrics[metrics["Segment"] == segment]
            rows.append(
                {
                    "Model": model_name,
                    "Segment": segment,
                    "users": len(group),
                    f"Precision@{k}": group["precision"].mean(),
                    f"Recall@{k}": group["recall"].mean(),
                    f"nDCG@{k}": group["ndcg"].mean(),
                    f"ILD@{k}": group["ild"].mean() if "ild" in group else np.nan,
                }
            )
    return pd.DataFrame(rows)
