"""Автономный пайплайн из src, работающий без ноутбуков."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import user_limit


@pytest.mark.parametrize(
    "value,expected", [("none", None), ("ALL", None), ("200", 200)]
)
def test_user_limits(value, expected):
    assert user_limit(value) == expected


@pytest.mark.parametrize("use_neural", [False, True])
def test_make_pipeline_end_to_end(tmp_path, use_neural):
    # В отдельной копии проекта есть только src и Makefile, без ноутбуков.
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "standalone project"
    project.mkdir()
    shutil.copytree(
        root / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(root / "Makefile", project / "Makefile")
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
            f"USE_NEURAL={int(use_neural)}",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = output / "reports"
    config = json.loads((reports / "config.json").read_text())
    assert config["k"] == 3
    assert config["evaluated_test_users"] == 8
    assert ("two_tower" in config["model_names"]) == use_neural
    metrics = pd.read_csv(reports / "metrics.csv")
    assert {"Hybrid (classical)", "Adaptive Hybrid + MMR"} <= set(metrics.Model)
    assert ("Hybrid + Two Tower" in set(metrics.Model)) == use_neural
    recs = pd.read_csv(reports / "recommendations.csv")
    assert set(recs.user_id) == {501, 9999}
    assert set(recs.book_id) <= set(range(101, 113))
    assert (reports / "report.md").is_file()
    assert (reports / "segments.png").is_file()
    assert (output / "processed" / "train.csv").is_file()
    assert result.stdout.count("1/6 Загрузка") == 1


def test_cli_uses_src_only():
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--notebook" not in result.stdout
    assert "--no-neural" in result.stdout
