"""Автономный запуск пайплайна; --notebook запускает указанный прототип."""

import argparse
import ast
import json
import sys
from pathlib import Path
from types import ModuleType

from src.config import PROJECT_DIR


def load_pipeline(notebook_path: str | Path):
    """Загрузить определения из доверенного ноутбука, пропустив примеры и выводы.

    Выполняются импорты, определения функций/классов и литеральные константы.
    Вызовы обучения, циклы экспериментов и сохранённые outputs не выполняются.
    Реализация алгоритмов остаётся в ноутбуке.
    """
    notebook_path = Path(notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    module = ModuleType("_book_notebook_pipeline")
    module.__file__ = str(notebook_path.resolve())
    module.PROJECT_ROOT = PROJECT_DIR
    sys.modules[module.__name__] = module

    import matplotlib

    matplotlib.use("Agg")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        tree = ast.parse(source, filename=f"{notebook_path}:cell-{index}")
        definitions = []
        for node in tree.body:
            if isinstance(
                node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
            ):
                # display нужен только интерактивным примерам, не run_pipeline.
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "IPython"
                ):
                    continue
                definitions.append(node)
            elif isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) for target in node.targets
            ):
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                definitions.append(node)
        code = ast.Module(body=definitions, type_ignores=[])
        exec(compile(code, f"{notebook_path}:cell-{index}", "exec"), module.__dict__)  # noqa: S102 - определения доверенного ноутбука
    if not callable(module.__dict__.get("run_pipeline")):
        raise TypeError(f"В {notebook_path} не определена функция run_pipeline")
    return module.run_pipeline


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
    parser.add_argument("--notebook", type=Path, default=None)
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
    notebook_path = args.pop("notebook")
    threads = args.pop("threads")
    args["use_neural"] = not args.pop("no_neural")
    if notebook_path is not None:
        run_pipeline = load_pipeline(notebook_path)
    else:
        import matplotlib

        matplotlib.use("Agg")
        from src.workflow import run_pipeline

    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        result = run_pipeline(**args)
    print(f"Готово. Отчёт: {result['report_path'].resolve()}")


if __name__ == "__main__":
    main()
