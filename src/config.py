"""Расположение всех тестовых данных, обучающих данных """

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


DATAFILE_LIST = ["books.csv", "ratings.csv", "book_tags.csv", "tags.csv", "to_read.csv"]

