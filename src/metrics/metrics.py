import numpy as np

def ranking_metrics(recommended, relevant, k=5):
    """Precision@K — число попаданий / K; Recall@K — число попаданий / число релевантных
        тестовых книг пользователя. nDCG@K учитывает позиции попаданий с весами
        `1 / log2(позиция + 1)` и делится на идеальный DCG.
        Общая функция ниже только считает метрики одного списка; рекомендации каждой
        модели получаем отдельно. В таблице усредняем метрики по пользователям."""
    hits = np.array([book_id in relevant for book_id in recommended[:k]], dtype=float)
    precision = hits.sum() / k
    recall = hits.sum() / len(relevant) if relevant else 0.0
    dcg = (hits / np.log2(np.arange(len(hits)) + 2)).sum()
    idcg = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
    return precision, recall, dcg / idcg if idcg > 0 else 0.0