FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV TEXTS_DB_PATH=/app/data/app.db \
    TEXTS_DB_DUMP_PATH=/app/data/backup.sql \
    TEXTS_DB_DUMP_INTERVAL_SEC=60 \
    TEXTS_TOP_K=5 \
    TEXTS_MARGIN_THRESHOLD=0.1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
