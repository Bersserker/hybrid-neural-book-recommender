"""Проверки соответствия автономных модулей текущему прототипу."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from notebook_reference import load_pipeline
from test_models import data as model_data_fixture

from src.data.features import ExtendedFeatureBuilder, build_feature_tables
from src.data.prepare_data import clean_ratings, train_test_val_per_user
from src.metrics.evaluation import evaluate_model
from src.metrics.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.metrics.segments import make_user_segments, summarize_segments
from src.models.adaptive_hybrid_recommender import (
    AdaptiveHybridRecommender,
    optimize_adaptive_weights,
)
from src.models.balanced_recommender import (
    BalancedRecommender,
    book_similarity,
    intra_list_diversity,
    mmr_rerank,
    tune_mmr,
)
from src.models.cb_item_recomennder import cb_item_recomennder
from src.models.content_based_recomender import content_based_recomender
from src.models.hybrid_recommender import HybridRecommender
from src.models.recommend_popular_recomemder import recommend_popular_recomemder
from src.models.svd_recommender import svd_recommender
from src.models.two_tower import TwoTower, TwoTowerRecommender, build_two_tower
from src.workflow import evaluate_pipeline_models, run_pipeline


@pytest.fixture
def data():
    return model_data_fixture.__wrapped__()


@pytest.fixture(scope="module")
def prototype():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "prototype.ipynb"
    return load_pipeline(path).__globals__


@pytest.mark.parametrize(
    "implementation",
    [
        clean_ratings,
        train_test_val_per_user,
        ExtendedFeatureBuilder,
        build_feature_tables,
        precision_at_k,
        recall_at_k,
        ndcg_at_k,
        evaluate_model,
        TwoTower,
        TwoTowerRecommender,
        HybridRecommender,
        AdaptiveHybridRecommender,
        optimize_adaptive_weights,
        BalancedRecommender,
        book_similarity,
        mmr_rerank,
        tune_mmr,
        intra_list_diversity,
        make_user_segments,
        summarize_segments,
        run_pipeline,
        evaluate_pipeline_models,
    ],
)
def test_definitions_match_notebook(prototype, implementation):
    # compile() ноутбука не регистрирует исходники в linecache.
    import json

    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks/prototype.ipynb").read_text()
    )
    expected = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name == implementation.__name__
            ):
                expected = node
    actual = ast.parse(inspect.getsource(implementation)).body[0]
    assert expected is not None
    assert ast.dump(actual) == ast.dump(expected)


@pytest.mark.parametrize(
    "name",
    [
        "recommend_popular_recomemder",
        "content_based_recomender",
        "cb_item_recomennder",
        "svd_recommender",
    ],
)
def test_classical_rankings_match(prototype, data, name):
    constructors = {
        "recommend_popular_recomemder": (
            recommend_popular_recomemder,
            [data["train_df"]],
        ),
        "content_based_recomender": (
            content_based_recomender,
            [data["train_df"], data["df_book_tags"], data["df_books"], data["df_tags"]],
        ),
        "cb_item_recomennder": (
            cb_item_recomennder,
            [data["train_df"], data["df_books"]],
        ),
        "svd_recommender": (svd_recommender, [data["train_df"], data["df_books"]]),
    }
    constructor, args = constructors[name]
    expected, actual = prototype[name](*args), constructor(*args)
    for user in [1, 2, 3, 999]:
        for size in [0, 1, 5, 20]:
            a, b = expected.recommend(user, size), actual.recommend(user, size)
            assert [r["book_id"] for r in a] == [r["book_id"] for r in b]
            np.testing.assert_allclose([r["score"] for r in a], [r["score"] for r in b])


def test_features_and_neural_training_match(prototype, data):
    builder, users, books = build_feature_tables(**data)
    ref_builder, ref_users, ref_books = prototype["build_feature_tables"](**data)
    loo = builder.transform(data["train_df"], leave_one_out=True)
    pd.testing.assert_frame_equal(users, ref_users)
    pd.testing.assert_frame_equal(books, ref_books)
    pd.testing.assert_frame_equal(
        loo, ref_builder.transform(data["train_df"], leave_one_out=True)
    )
    expected_counts = (
        data["train_df"]["user_id"].map(data["train_df"].groupby("user_id").size()) - 1
    )
    np.testing.assert_array_equal(loo["user_ratings_count"], expected_counts)
    torch.set_num_threads(1)
    actual = build_two_tower(**data, embedding_dim=8, output_dim=4, device="cpu")
    expected = prototype["TwoTowerRecommender"](
        data["train_df"],
        data["df_books"],
        train_features=loo,
        user_features=users,
        book_features=books,
        embedding_dim=8,
        output_dim=4,
        device="cpu",
    )
    relevant = {1: {3}, 2: {5}, 3: {1}}
    actual.fit(relevant, epochs=2, batch_size=2)
    expected.fit(relevant, epochs=2, batch_size=2)
    pd.testing.assert_frame_equal(actual.history, expected.history)
    for user in [1, 2, 3, 999]:
        assert actual.recommend(user) == expected.recommend(user)
    models = {
        "two_tower": actual,
        "pop": recommend_popular_recomemder(data["train_df"]),
    }
    hybrid = HybridRecommender(data["train_df"], models)
    adaptive = AdaptiveHybridRecommender(data["train_df"], models)
    assert hybrid.default_weights["two_tower"] == pytest.approx(0.2 / 0.28)
    assert set(adaptive.adaptive_weights["new"]) == set(models)
    for cls, model in [
        ("HybridRecommender", hybrid),
        ("AdaptiveHybridRecommender", adaptive),
    ]:
        reference = prototype[cls](data["train_df"], models)
        assert model.recommend(1) == reference.recommend(1)
