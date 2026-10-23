"""Проверки вспомогательного загрузчика эталона для тестов."""

import json

import pytest
from notebook_reference import load_pipeline


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
