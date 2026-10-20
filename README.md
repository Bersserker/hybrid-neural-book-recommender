# Гибридная система рекомендаций книг

Прототип: [notebooks/hybrid_book_recommender_prototype.ipynb](notebooks/hybrid_book_recommender_prototype.ipynb).
Разделы следуют нумерации `task.md`, код и комментарии к моделям находятся в ноутбуке.

## Запуск

1. Установите Python 3.12+ и зависимости: `make install`.
2. Проверьте наличие `books.csv`, `ratings.csv`, `tags.csv`, `book_tags.csv` в `data/raw/`.
3. Запустите `uv run jupyter lab`, откройте ноутбук и выполните **Restart Kernel → Run All**.

По умолчанию используются 6000 случайных пользователей, до 150 пользователей
каждого сегмента для оценки и 5 эпох обучения на CPU. Параметры находятся в
`CONFIG` в начале ноутбука. Для полной выборки задайте `max_users=None` и
`eval_per_segment=None`; время выполнения и потребление памяти вырастут.

## Решение

Подготовка данных для пайплайна из `notebooks/prototype.ipynb`:

```bash
uv run python -m src.data.prepare_data
```

Команда создаёт `data/processed/` и сохраняет очищенные `ratings.csv`,
выборки `train.csv`, `val.csv`, `test.csv` (60/20/20 по пользователям,
`random_state=42`), а также `user_history.pkl`, `val_relevant.pkl` и
`test_relevant.pkl`. Словари загружаются через `pickle.load` и сохраняют
структуру `user_id → set(book_id)`. История содержит только train;
релевантные книги имеют оценку ≥ 4. Повторный запуск перезаписывает файлы.
Для вызова из Python доступна `prepare_data(raw_data_dir, processed_data_dir)`,
возвращающая словарь путей к сохранённым файлам.

### Обучение моделей из `prototype.ipynb`

После подготовки данных все модели обучаются командой:

```bash
uv run python -m src.models.train --model all --epochs 5 --n-trials 30
```

Для отдельной модели укажите `--model`:

| Значение | Класс | Модуль в `src/models/` |
| --- | --- | --- |
| `pop` | `recommend_popular_recomemder` | `recommend_popular_recomemder.py` |
| `content` | `content_based_recomender` | `content_based_recomender.py` |
| `item` | `cb_item_recomennder` | `cb_item_recomennder.py` |
| `svd` | `svd_recommender` (Surprise SVD) | `svd_recommender.py` |
| `hybrid` | `HybridRecommender` | `hybrid_recommender.py` |
| `adaptive` | `AdaptiveHybridRecommender` | `adaptive_hybrid_recommender.py` |
| `two_tower` | `TwoTower`, `TwoTowerRecommender` | `two_tower.py` |
| `balanced` | `BalancedRecommender` (адаптивный гибрид + MMR) | `balanced_recommender.py` |

Каждый модуль также запускается отдельно через `python -m src.models.<модуль>`
и предоставляет функцию `train_*_recommender()` для вызова из Python.
В режиме `all` гибриды переиспользуют обученные базовые модели.
Two Tower получает ID-эмбеддинги и числовые признаки из `src/data/features.py`;
при обучении статистики вычисляются leave-one-out, при выдаче — по полному train.
Гибриды включают Two Tower, `balanced` добавляет MMR с подбором на validation.
Обучение использует `data/processed/train.csv` и таблицы каталога из `data/raw/`.
Адаптивные веса и лучшая эпоха Two Tower выбираются по `val_relevant.pkl`;
test при обучении не читается. `--n-trials 0` сохраняет исходные адаптивные веса.
`--device cpu` задаёт обучение Two Tower на CPU.

Артефакты сохраняются в `artifacts/*.joblib`, история Two Tower — в
`two_tower_history.csv`, результаты подбора весов — в `adaptive_weight_search.csv`
в той же папке. Повторный запуск перезаписывает файлы. Пути можно изменить
параметрами `--processed-data-dir`, `--raw-data-dir`, `--models-dir`.

```python
import joblib

model = joblib.load("artifacts/adaptive_hybrid_recommender.joblib")
recommendations = model.recommend(user_id=1, N=10)
```

Все модели возвращают список `{"book_id": int, "score": float}` и исключают
прочитанные в train книги. Two Tower сохраняется на CPU, чтобы артефакт можно
было загрузить без CUDA.

### Команды Make и отчёты

Обучение и отчёты доступны через Make:

```bash
make train                       # обучить все модели
make train-content               # обучить только Content-Based
make train-two_tower EPOCHS=10 DEVICE=cpu
make train-adaptive N_TRIALS=50
make report                      # отчёт по всем сохранённым моделям на val
make report-pop K=20             # отчёт только по Popularity
make report SPLIT=test           # итоговая оценка на test
make train-report                # последовательно обучить и создать отчёт
make train-report MODEL=svd      # тот же процесс для одной модели
```

Суффиксы `train-` и `report-`: `pop`, `content`, `item`, `svd`, `hybrid`,
`adaptive`, `two_tower`. `make train` предполагает, что данные уже подготовлены
через `make prepare_data`. `make report` требует наличия всех семи артефактов;
если обучена одна модель, используйте соответствующий `report-<модель>`.

Настройки: `MODEL=all`, `EPOCHS=5`, `N_TRIALS=30`, `K=10`, `SPLIT=val`;
`DEVICE` по умолчанию выбирается автоматически. Пути переопределяются через
`PROCESSED_DATA_DIR`, `RAW_DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`.
Например: `make train-report MODEL=pop REPORTS_DIR=experiment_reports`.

Отчёты сохраняются в `reports/<val|test>/<модель|all>/`: `report.md`,
сводная таблица `metrics.csv`, график `metrics.png` и метрики по пользователям
в `<модель>_users.csv`. Для Two Tower добавляются история и график обучения,
для адаптивного гибрида — таблицы весов и результатов Optuna.
Метрики усредняются по всем пользователям с релевантными книгами.
Рядом с моделями новые запуски обучения через Make или `src.models.train`
сохраняют `*.training.json` с временем обучения и параметрами запуска.
Старые артефакты также поддерживаются, но время их обучения не восстанавливается.
Повторный запуск обновляет отчёт для выбранной выборки и модели.

### Сквозной запуск актуального прототипа через Make

```bash
make install                      # установить зависимости в .venv
make pipeline-smoke               # проверочный запуск на 200 пользователях
make pipeline                     # полный запуск с Two Tower, гибридом и MMR
make pipeline-classical           # тот же процесс без нейросети
```

Команды `pipeline*` используют автономную `run_pipeline()` из `src/workflow.py`:
загрузка CSV → очистка и разбиение → расширенные признаки → обучение → подбор
весов и MMR на validation → выдача рекомендаций → test-оценка и отчёт по сегментам.
Реализация моделей находится в `src/models/`, подготовка признаков — в
`src/data/features.py`, анализ сегментов — в `src/metrics/segments.py`.
Для запуска непосредственно прототипа укажите `make pipeline NOTEBOOK=notebooks/prototype.ipynb`.
Команды `train` и `report` выше запускают отдельные модули `src.models` и `src.metrics`.

```bash
make pipeline MAX_USERS=1000 EPOCHS=5 N_TRIALS=30 DEVICE=cpu
make pipeline RAW_DATA_DIR=/path/to/csv PIPELINE_OUTPUT_DIR=artifacts/experiment
make pipeline MAX_USERS=200 RECOMMEND_USERS=1,2,3
make pipeline DOWNLOAD=1          # сначала скачать Goodbooks
```

Параметры полного запуска: `MAX_USERS=none`, `TUNING_USERS=200`, `EVAL_USERS=none`,
`EPOCHS=5`, `SVD_EPOCHS=50`, `N_TRIALS=30`, `BATCH_SIZE=1024`, `K=10`,
`CANDIDATES_K=100`, `RANDOM_STATE=42`, `THREADS=2`, `USE_NEURAL=1`, `DOWNLOAD=0`.
`none` или `all` отключают ограничение числа пользователей. `DEVICE` выбирается
автоматически. `RECOMMEND_USERS` — ID через запятую; по умолчанию сохраняются
примеры для первых пяти оцениваемых test-пользователей.

`pipeline-smoke` задаёт 200 пользователей, 2 нейросетевые эпохи, 5 эпох SVD,
3 trials на сегмент, 100 validation-пользователей и всех подходящих test-пользователей
этой подвыборки. Это проверка всей цепочки, а не итоговый эксперимент на полной базе.

По умолчанию результаты находятся в `artifacts/notebook_pipeline/`:
`processed/{train,val,test}.csv`, `reports/report.md`, общие и сегментные метрики,
рекомендации с названиями книг, графики, история Two Tower, выбранные веса и параметры.
Для проверочного запуска папка заканчивается на `_smoke`, для классического —
на `_classical`. Изменить исходный путь можно через `PIPELINE_OUTPUT_DIR`.
Повторный запуск обновляет результаты в выбранной папке.

Make использует установленное окружение через `uv run --no-sync`; после изменения
зависимостей выполните `make install`. Установка выполняется с `--no-install-project`,
поскольку команды запускают модули напрямую из репозитория. При необходимости
интерпретатор можно задать явно: `make pipeline-smoke PYTHON=.venv/bin/python`.

### Описание исходного прототипа

- Признаки пользователей, книг, жанров и близости к истории считаются по train;
  общедоступные названия и теги доступны для всего каталога.
- Базовые модели: Popularity, Content-Based TF-IDF, Item-Based CF и TruncatedSVD
  (упрощённая факторизация, отличающаяся от Surprise SVD из домашнего задания).
- Гибрид объединяет кандидатов, фильтрует прочитанное, взвешивает прогнозы
  отдельно для новых и активных пользователей и добавляет жанровое разнообразие.
- Веса и баланс разнообразия выбираются по validation. Test используется
  для сравнения Precision@K, Recall@K, nDCG@K и анализа сегментов.
- Wide & Deep объединяет ID-эмбеддинги и расширенные признаки и включается
  в общий гибрид. Неизвестным пользователям выдаются популярные книги.
- `run_pipeline(data_dir, cfg, use_neural=True)` повторяет весь процесс для
  другой папки с CSV той же схемы. `recommend_books` возвращает итоговую выдачу.

В ноутбуке есть графики, таблицы, проверки корректности выдачи и выводы,
которые используют фактические результаты запуска. Ограничения прототипа
и направления дальнейших экспериментов описаны в разделе 5.

Исходные ноутбуки `Teplayakov_AN_HOME1.ipynb` и `eda_2.ipynb` сохранены.
