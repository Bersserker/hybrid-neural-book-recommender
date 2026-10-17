"""Загрузка исходного датасета и подготовка данных."""

from collections.abc import Sequence
from pathlib import Path

import kagglehub

from src.config import DATAFILE_LIST, RAW_DATA_DIR

DATASET = "zygmunt/goodbooks-10k"


def load_kaggle_dataset(
    path: str | Path = RAW_DATA_DIR,
    files_list: Sequence[str] | None = None,
    *,
    force_download: bool = False,
) -> dict[str, Path]:
    """Скачать датасет и вернуть пути к проверенным исходным файлам.

    По умолчанию используются папка data/raw/ и файлы из DATAFILE_LIST.
    files_list задаёт файлы для проверки и возврата, а не состав загрузки.
    Если ожидаемые файлы отсутствуют, возбуждается FileNotFoundError.
    """
    if files_list is None:
        files_list = DATAFILE_LIST
    path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    kagglehub.dataset_download(
        DATASET, output_dir=str(path), force_download=force_download
    )

    dataset_path = {file: path / file for file in files_list}
    missing_files = [
        str(file_path) for file_path in dataset_path.values() if not file_path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Отсутствуют ожидаемые файлы датасета: " + ", ".join(missing_files)
        )
    return dataset_path


if __name__ == "__main__":
    load_kaggle_dataset()
