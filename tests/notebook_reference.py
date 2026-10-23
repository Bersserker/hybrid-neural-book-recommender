"""Загрузка прототипа только для тестов соответствия реализаций."""

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
