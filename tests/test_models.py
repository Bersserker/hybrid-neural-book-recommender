"""Проверки переноса моделей из ноутбука в автономный пайплайн."""

import json
import pickle
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.adaptive_hybrid_recommender import (
    AdaptiveHybridRecommender,
    optimize_adaptive_weights,
)
from src.models.cb_item_recomennder import cb_item_recomennder
from src.models.content_based_recomender import content_based_recomender
from src.models.hybrid_recommender import HybridRecommender, build_base_models
from src.models.svd_recommender import svd_recommender
from src.models.train import MODEL_FILES
from src.models.two_tower import TwoTower, TwoTowerRecommender


@pytest.fixture
def data():
    train = pd.DataFrame(
        [
            (1, 1, 5),
            (1, 2, 1),
            (2, 1, 4),
            (2, 3, 5),
            (2, 4, 2),
            (3, 3, 4),
            (3, 5, 5),
            (3, 6, 1),
        ],
        columns=["user_id", "book_id", "rating"],
    )
    titles = [
        "space adventure",
        "romantic poetry",
        "space adventure",
        "ancient history",
        "space galaxy",
        "cooking food",
        "new story",
        "travel",
    ]
    books = pd.DataFrame(
        {
            "id": range(1, 9),
            "book_id": range(101, 109),
            "title": titles,
            "original_title": titles,
            "authors": ["Author"] * 8,
        }
    )
    book_tags = pd.DataFrame(
        {
            "goodreads_book_id": range(101, 109),
            "tag_id": [1, 2, 1, 3, 1, 4, 5, 5],
            "count": [10] * 8,
        }
    )
    tags = pd.DataFrame(
        {
            "tag_id": range(1, 6),
            "tag_name": ["science-fiction", "poetry", "history", "food", "new"],
        }
    )
    return {
        "train_df": train,
        "df_books": books,
        "df_book_tags": book_tags,
        "df_tags": tags,
    }


def test_content_profile_uses_positive_train_only(data):
    model = content_based_recomender(**data)
    assert model.positive_history[1] == [1]
    assert model.recommend(1, 1)[0]["book_id"] == 3
    assert model.recommend(999) == model.recommend_from_top(999)


def test_item_cf_matches_weighted_cosine(data):
    model = cb_item_recomennder(data["train_df"], data["df_books"])
    assert np.allclose(model.item_similarity.diagonal(), 0)
    recs = model.recommend(1)
    ratings = model.interaction_matrix.getrow(model.user_to_idx[1]).toarray().ravel()
    sim = model.item_similarity.toarray()
    for rec in recs:
        idx = model.book_to_idx[rec["book_id"]]
        observed = ratings > 0
        expected = (ratings[observed] @ sim[observed, idx]) / sim[observed, idx].sum()
        assert rec["score"] == pytest.approx(expected)
    assert model.recommend(999) == model.recommend_from_top(999)


def test_svd_vector_scores_match_surprise_predict(data):
    model = svd_recommender(data["train_df"], data["df_books"], n_epochs=2)
    for user_id in [1, 999]:
        for rec in model.recommend(user_id, N=20):
            expected = model.svd_model.predict(user_id, rec["book_id"]).est
            assert rec["score"] == pytest.approx(expected)


@pytest.fixture
def base_models(data):
    return build_base_models(**data, svd_kwargs={"n_epochs": 2})


@pytest.mark.parametrize("name", ["content", "item", "svd", "hybrid", "adaptive"])
def test_recommendation_contract(data, base_models, name):
    models = dict(base_models)
    models["hybrid"] = HybridRecommender(data["train_df"], models=base_models)
    models["adaptive"] = AdaptiveHybridRecommender(data["train_df"], models=base_models)
    model = models[name]
    for user_id in [1, 999]:
        recs = model.recommend(user_id, N=20)
        ids = [r["book_id"] for r in recs]
        assert len(ids) == len(set(ids))
        assert set(ids) <= set(data["df_books"]["id"])
        assert not set(ids) & model.user_history.get(user_id, set())
        assert all(np.isfinite(r["score"]) for r in recs)
        assert all(a["score"] >= b["score"] for a, b in pairwise(recs))
        assert model.recommend(user_id, N=0) == []
        assert model.recommend(user_id, N=-1) == []


def test_hybrid_weighting_and_validation(data, base_models):
    model = HybridRecommender(data["train_df"], models=base_models)
    weights = {"pop": 1, "content": 0, "item": 0, "svd": 0}
    normalized = model.normalize_scores(base_models["pop"].recommend(1, N=100))
    result = model.recommend(1, weights=weights)
    assert all(rec["score"] == normalized.get(rec["book_id"], 0) for rec in result)
    assert model.normalize_scores([]) == {}
    assert model.normalize_scores([{"book_id": 1, "score": 3}]) == {1: 1}
    for invalid in [
        {"pop": 1},
        dict(weights, pop=-1, svd=2),
        dict(weights, pop=2),
        dict(weights, pop=float("nan")),
    ]:
        with pytest.raises(ValueError):
            model.recommend(1, weights=invalid)


def test_adaptive_groups_and_optimization(data, base_models):
    train = pd.DataFrame(
        {
            "user_id": [1] * 4 + [2] * 5 + [3] * 50,
            "book_id": [1] * 59,
            "rating": [5] * 59,
        }
    )
    model = AdaptiveHybridRecommender(train, models=base_models)
    assert [model.get_user_type(u) for u in [999, 1, 2, 3]] == [
        "new",
        "new",
        "regular",
        "active",
    ]
    best, results = optimize_adaptive_weights(
        model, {1: {3}, 2: {5}, 3: {1}}, n_trials=1
    )
    assert set(results["user_type"]) == {"new", "regular", "active"}
    for weights in best.values():
        assert sum(weights.values()) == pytest.approx(1)
        assert all(w >= 0 for w in weights.values())


def test_two_tower_best_epoch_restoration(data, monkeypatch):
    model = TwoTowerRecommender(
        data["train_df"],
        data["df_books"],
        use_features=False,
        embedding_dim=8,
        output_dim=4,
        device="cpu",
    )
    assert isinstance(model.network, TwoTower)
    original = {k: v.clone() for k, v in model.network.state_dict().items()}
    states = []

    def evaluate(*args, **kwargs):
        states.append({k: v.clone() for k, v in model.network.state_dict().items()})
        return pd.DataFrame({"ndcg": [1 if len(states) == 1 else 0]})

    monkeypatch.setattr("src.models.two_tower.evaluate_model", evaluate)
    model.fit({1: {3}}, epochs=2, batch_size=2)
    assert model.best_epoch == 1
    assert len(model.history) == 2
    assert any(not torch.equal(v, original[k]) for k, v in states[0].items())
    assert all(
        torch.equal(v, states[0][k]) for k, v in model.network.state_dict().items()
    )
    for user_id in [1, 999]:
        recs = model.recommend(user_id, N=20)
        ids = {r["book_id"] for r in recs}
        assert not ids & model.user_history.get(user_id, set())
        assert ids <= set(data["train_df"]["book_id"])
    assert model.recommend(1, N=-1) == []


def test_cli_all_and_artifact_loading_in_another_process(data, tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    raw.mkdir()
    processed.mkdir()
    data["train_df"].to_csv(processed / "train.csv", index=False)
    for name in ["books", "book_tags", "tags"]:
        data[f"df_{name}"].to_csv(raw / f"{name}.csv", index=False)
    with (processed / "val_relevant.pkl").open("wb") as file:
        pickle.dump({1: {3}, 2: {5}, 3: {1}}, file)
    # Отсутствие test-файлов также проверяет, что обучение не читает test.
    result = subprocess.run(
        [
            "make",
            "train-report",
            f"PYTHON={sys.executable}",
            f"RAW_DATA_DIR={raw}",
            f"PROCESSED_DATA_DIR={processed}",
            f"MODELS_DIR={artifacts}",
            f"REPORTS_DIR={tmp_path / 'reports'}",
            "EPOCHS=1",
            "N_TRIALS=1",
            "DEVICE=cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for filename in MODEL_FILES.values():
        model = joblib.load(artifacts / filename)
        assert isinstance(model.recommend(1), list)
        assert model.recommend(999)
        metadata = json.loads(
            (artifacts / filename).with_suffix(".training.json").read_text()
        )
        assert metadata["duration_seconds"] > 0
    assert len(pd.read_csv(artifacts / "two_tower_history.csv")) == 1
    assert not pd.read_csv(artifacts / "adaptive_weight_search.csv").empty
    report_dir = tmp_path / "reports" / "val" / "all"
    summary = pd.read_csv(report_dir / "metrics.csv")
    assert set(summary["model"]) == set(MODEL_FILES)
    assert (summary["users"] == 3).all()
    assert summary[["Precision@10", "Recall@10", "nDCG@10"]].ge(0).all().all()
    assert summary[["Precision@10", "Recall@10", "nDCG@10"]].le(1).all().all()
    for filename in [
        "report.md",
        "metrics.png",
        "two_tower_training.png",
        "adaptive_weights.csv",
    ]:
        assert (report_dir / filename).stat().st_size > 0
