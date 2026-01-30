# Texts Annotation

Streamlit-приложение для разметки текстов с учетом требований:

- Мульти-лейбл классификация с ответами **yes/no/unsure** по каждому кандидату.
- Разметка минимум двумя разметчиками (счетчик хранится в БД).
- Отбор topK классов по вероятностям без порогов.
- Сохранение в БД версии модели, которая сформировала кандидатов.
- Автодамп БД на диск и восстановление при перезапуске.
- Заглушка для обучения новой версии модели на накопленных данных.
- YAML-словарь интентов с описанием, сложностью и кластером.

## Запуск локально

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Запуск в Docker

```bash
docker build -t texts-annotation .
docker run --rm -p 8501:8501 -v $PWD/data:/app/data texts-annotation
```

## Переменные окружения

- `TEXTS_DB_PATH` — путь до SQLite БД (по умолчанию `data/app.db`).
- `TEXTS_DB_DUMP_PATH` — путь до дампа (по умолчанию `data/backup.sql`).
- `TEXTS_DB_DUMP_INTERVAL_SEC` — интервал автодампа в секундах (по умолчанию `60`).
- `TEXTS_TOP_K` — размер topK (по умолчанию `5`).
- `TEXTS_MARGIN_THRESHOLD` — порог для `margin_error_rate`.
- `TEXTS_INTENTS_PATH` — путь к YAML-словарю интентов или к папке с несколькими YAML файлами.

## Структура БД

- `texts` — тексты, метаданные (язык, кластеры), версия данных.
- `candidates` — topK кандидаты + вероятность и версия модели.
- `annotations` — выборы разметчиков (включая метки вне topK).
- `model_versions` — версии моделей.
- `settings` — текущие версии и настройки.
