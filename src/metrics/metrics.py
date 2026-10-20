import numpy as np


def precision_at_k(actual, predicted, k=10):
    if k <= 0:
        return 0.0

    actual = set(actual)
    predicted = list(dict.fromkeys(predicted))[:k]

    hits = len(actual.intersection(predicted))

    return hits / k


def recall_at_k(actual, predicted, k=10):
    actual = set(actual)

    if not actual or k <= 0:
        return 0.0

    predicted = list(dict.fromkeys(predicted))[:k]

    hits = len(actual.intersection(predicted))

    return hits / len(actual)


def ndcg_at_k(actual, predicted, k=10):
    actual = set(actual)

    if not actual or k <= 0:
        return 0.0

    predicted = list(dict.fromkeys(predicted))[:k]

    relevance = np.array([1 if book_id in actual else 0 for book_id in predicted])

    positions = np.arange(1, len(relevance) + 1)

    dcg = np.sum(relevance / np.log2(positions + 1))

    ideal_hits = min(len(actual), k)

    ideal_positions = np.arange(1, ideal_hits + 1)

    idcg = np.sum(1 / np.log2(ideal_positions + 1))

    return float(dcg / idcg)
