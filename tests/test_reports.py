"""Проверки метрик отчёта и Make-команд для отдельных моделей."""

import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.metrics.report import create_report
from src.models.recommend_popular_recomemder import train_popular_recommender


@pytest.fixture
def saved_pop(tmp_path):
    processed = tmp_path / "processed data"
    artifacts = tmp_path / "model artifacts"
    processed.mkdir()
    pd.DataFrame(
        [(1, 1, 5), (2, 2, 4)], columns=["user_id", "book_id", "rating"]
    ).to_csv(processed / "train.csv", index=False)
    train_popular_recommender(processed, artifacts / "popular_recommender.joblib")
    for split, relevant in [("val", {1: {2}, 999: {1}}), ("test", {1: {3}})]:
        with (processed / f"{split}_relevant.pkl").open("wb") as file:
            pickle.dump(relevant, file)
    return processed, artifacts


def test_report_metrics_and_legacy_artifact(saved_pop, tmp_path):
    processed, artifacts = saved_pop
    paths = create_report(
        "pop",
        k=2,
        processed_data_dir=processed,
        models_dir=artifacts,
        reports_dir=tmp_path / "reports",
    )
    metrics = pd.read_csv(paths["metrics"]).iloc[0]
    assert metrics["users"] == 2
    assert metrics["Precision@2"] == 0.5
    assert metrics["Recall@2"] == metrics["nDCG@2"] == 1
    assert pd.isna(metrics["training_seconds"])
    assert all(path.is_file() for path in paths.values())


def test_make_individual_report_uses_requested_test_split(saved_pop, tmp_path):
    processed, artifacts = saved_pop
    result = subprocess.run(
        [
            "make",
            "report-pop",
            f"PYTHON={sys.executable}",
            f"PROCESSED_DATA_DIR={processed}",
            f"MODELS_DIR={artifacts}",
            f"REPORTS_DIR={tmp_path / 'reports'}",
            "SPLIT=test",
            "K=2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = pd.read_csv(tmp_path / "reports" / "test" / "pop" / "metrics.csv").iloc[0]
    assert metrics["users"] == 1
    assert metrics["nDCG@2"] == 0


def test_report_fails_clearly_for_missing_models_and_empty_validation(
    saved_pop, tmp_path
):
    processed, artifacts = saved_pop
    with pytest.raises(FileNotFoundError, match="Сначала обучите"):
        create_report(models_dir=artifacts, processed_data_dir=processed)
    with (processed / "val_relevant.pkl").open("wb") as file:
        pickle.dump({}, file)
    with pytest.raises(ValueError, match="непустая выборка"):
        create_report("pop", models_dir=artifacts, processed_data_dir=processed)
    with pytest.raises(ValueError, match="k >= 1"):
        create_report("pop", k=0)
