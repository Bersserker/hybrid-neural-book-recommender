import pandas as pd

from src.metrics.metrics import ndcg_at_k, precision_at_k, recall_at_k


def evaluate_model(recommend_fn, relevant, k=10, verbose=False):
    from time import perf_counter

    results = []
    started = perf_counter()

    for index, (user_id, actual) in enumerate(relevant.items(), start=1):
        raw_predicted = recommend_fn(user_id, N=k)
        predicted = [rec["book_id"] for rec in raw_predicted]

        results.append(
            {
                "user_id": user_id,
                "precision": precision_at_k(actual, predicted, k),
                "recall": recall_at_k(actual, predicted, k),
                "ndcg": ndcg_at_k(actual, predicted, k),
            }
        )

        if verbose and (index % 500 == 0 or index == len(relevant)):
            elapsed = perf_counter() - started
            eta = elapsed / index * (len(relevant) - index)
            print(
                f"{index}/{len(relevant)} пользователей | {elapsed:.1f} с | осталось ≈ {eta:.1f} с",
                flush=True,
            )

    return pd.DataFrame(
        results,
        columns=["user_id", "precision", "recall", "ndcg"],
    )
