"""Запуск сквозного пайплайна из модулей src."""

import argparse
from pathlib import Path

from src.config import PROJECT_DIR


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Нужно положительное целое число")
    return number


def user_limit(value):
    return None if value.lower() in {"none", "all"} else positive_int(value)


def user_ids(value):
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Укажите ID через запятую: 1,2,3") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "notebook_pipeline",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--max-users", type=user_limit, default=None)
    parser.add_argument("--tuning-users", type=user_limit, default=200)
    parser.add_argument("--eval-users", type=user_limit, default=None)
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=1024)
    parser.add_argument("--svd-epochs", type=positive_int, default=50)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--candidates-k", type=positive_int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--recommend-user-ids", type=user_ids, default=None)
    args = vars(parser.parse_args())
    threads = args.pop("threads")
    args["use_neural"] = not args.pop("no_neural")
    import matplotlib

    matplotlib.use("Agg")
    import torch
    from threadpoolctl import threadpool_limits

    from src.workflow import run_pipeline

    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        result = run_pipeline(**args)
    print(f"Готово. Отчёт: {result['report_path'].resolve()}")


if __name__ == "__main__":
    main()
