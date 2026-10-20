"""Автономный пайплайн и опциональная загрузка прототипа."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import load_pipeline, user_limit


@pytest.mark.parametrize(
    "value,expected", [("none", None), ("ALL", None), ("200", 200)]
)
def test_user_limits(value, expected):
    assert user_limit(value) == expected


def test_loader_skips_examples_and_training_calls(tmp_path):
    marker = tmp_path / "example-ran"
    notebook = tmp_path / "prototype.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "from pathlib import Path\n",
                            "FACTOR = 2\n",
                            "def run_pipeline(value=3):\n    return value * FACTOR\n",
                            f"result = Path({str(marker)!r}).write_text('example')\n",
                            "for i in range(1):\n    raise RuntimeError('example loop')\n",
                        ],
                    }
                ]
            }
        )
    )
    pipeline = load_pipeline(notebook)
    assert not marker.exists()
    assert pipeline() == 6
    assert pipeline(value=5) == 10


def test_missing_pipeline_is_reported(tmp_path):
    notebook = tmp_path / "empty.ipynb"
    notebook.write_text(json.dumps({"cells": []}))
    with pytest.raises(TypeError, match="run_pipeline"):
        load_pipeline(notebook)


@pytest.mark.parametrize("notebook", [False, True])
def test_make_pipeline_end_to_end(tmp_path, notebook):
    raw = tmp_path / "raw files"
    output = tmp_path / "pipeline output"
    raw.mkdir()
    pd.DataFrame(
        {
            "id": range(101, 113),
            "book_id": range(1001, 1013),
            "title": [f"space adventure {i}" for i in range(12)],
            "original_title": [f"space adventure {i}" for i in range(12)],
            "authors": ["Author"] * 12,
        }
    ).to_csv(raw / "books.csv", index=False)
    rows = [
        (user, book, 4 + (user + book) % 2)
        for user in range(501, 509)
        for book in range(101, 111)
    ]
    pd.DataFrame(rows, columns=["user_id", "book_id", "rating"]).to_csv(
        raw / "ratings.csv", index=False
    )
    pd.DataFrame(
        {"goodreads_book_id": range(1001, 1013), "tag_id": [1] * 12, "count": [10] * 12}
    ).to_csv(raw / "book_tags.csv", index=False)
    pd.DataFrame({"tag_id": [1], "tag_name": ["science-fiction"]}).to_csv(
        raw / "tags.csv", index=False
    )
    pd.DataFrame(columns=["user_id", "book_id"]).to_csv(
        raw / "to_read.csv", index=False
    )
    result = subprocess.run(
        [
            "make",
            "pipeline",
            f"PYTHON={sys.executable}",
            f"RAW_DATA_DIR={raw}",
            f"PIPELINE_OUTPUT_DIR={output}",
            "EPOCHS=1",
            "SVD_EPOCHS=1",
            "N_TRIALS=1",
            "BATCH_SIZE=32",
            "K=3",
            "CANDIDATES_K=10",
            "TUNING_USERS=none",
            "EVAL_USERS=none",
            "THREADS=1",
            "DEVICE=cpu",
            "RECOMMEND_USERS=501,9999",
            "NOTEBOOK=notebooks/prototype.ipynb" if notebook else "NOTEBOOK=",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = output / "reports"
    config = json.loads((reports / "config.json").read_text())
    assert config["k"] == 3
    assert config["evaluated_test_users"] == 8
    assert "two_tower" in config["model_names"]
    metrics = pd.read_csv(reports / "metrics.csv")
    assert {"Hybrid + Two Tower", "Hybrid (classical)", "Adaptive Hybrid + MMR"} <= set(
        metrics.Model
    )
    recs = pd.read_csv(reports / "recommendations.csv")
    assert set(recs.user_id) == {501, 9999}
    assert set(recs.book_id) <= set(range(101, 113))
    assert (reports / "report.md").is_file()
    assert (reports / "segments.png").is_file()
    assert (output / "processed" / "train.csv").is_file()
    assert result.stdout.count("1/6 Загрузка") == 1
    if not notebook:
        # Сравниваем весь автономный процесс с текущим прототипом на тех же CSV.
        from src.workflow import run_pipeline

        reference = load_pipeline(
            Path(__file__).resolve().parents[1] / "notebooks/prototype.ipynb"
        )
        kwargs = {
            "data_dir": raw,
            "epochs": 1,
            "svd_epochs": 1,
            "n_trials": 1,
            "batch_size": 32,
            "k": 3,
            "candidates_k": 10,
            "tuning_users": None,
            "eval_users": None,
            "device": "cpu",
            "recommend_user_ids": [501, 9999],
        }
        expected = reference(output_dir=tmp_path / "reference", **kwargs)
        actual = run_pipeline(output_dir=tmp_path / "standalone", **kwargs)
        for key in ["metrics", "segments", "recommendations"]:
            pd.testing.assert_frame_equal(actual[key], expected[key])
        assert (
            actual["adaptive"].adaptive_weights == expected["adaptive"].adaptive_weights
        )
        assert actual["model"].relevance_weight == expected["model"].relevance_weight
