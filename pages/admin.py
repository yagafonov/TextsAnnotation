import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import streamlit as st

from db import connect

ADMIN_PASSWORD = os.environ.get("TEXTS_ADMIN_PASSWORD", "admin123")

st.set_page_config(page_title="Admin Dashboard", layout="wide")

st.title("Панель администратора")

with st.sidebar:
    st.header("Авторизация")
    admin_password = st.text_input("Пароль администратора", type="password")
    if admin_password != ADMIN_PASSWORD:
        st.warning("Введите пароль администратора.")
        st.stop()
    st.success("Авторизация успешна")


def load_overview_stats() -> Dict:
    with connect() as conn:
        total_texts = conn.execute("SELECT COUNT(*) as cnt FROM texts").fetchone()["cnt"]
        texts_with_annotations = conn.execute(
            "SELECT COUNT(DISTINCT text_id) as cnt FROM annotations"
        ).fetchone()["cnt"]
        total_annotations = conn.execute("SELECT COUNT(*) as cnt FROM annotations").fetchone()["cnt"]
        unique_annotators = conn.execute(
            "SELECT COUNT(DISTINCT annotator) as cnt FROM annotations"
        ).fetchone()["cnt"]
        texts_by_cluster = conn.execute(
            """
            SELECT assigned_cluster, COUNT(*) as cnt
            FROM texts
            GROUP BY assigned_cluster
            """
        ).fetchall()
    return {
        "total_texts": total_texts,
        "texts_with_annotations": texts_with_annotations,
        "pending_texts": total_texts - texts_with_annotations,
        "total_annotations": total_annotations,
        "unique_annotators": unique_annotators,
        "texts_by_cluster": {row["assigned_cluster"] or "unknown": row["cnt"] for row in texts_by_cluster},
    }


def load_annotator_stats() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                annotator,
                COUNT(*) as total_annotations,
                COUNT(DISTINCT text_id) as texts_annotated,
                SUM(CASE WHEN decision = 'yes' THEN 1 ELSE 0 END) as yes_count,
                SUM(CASE WHEN decision = 'no' THEN 1 ELSE 0 END) as no_count,
                SUM(CASE WHEN decision = 'unsure' THEN 1 ELSE 0 END) as unsure_count,
                MIN(created_at) as first_annotation,
                MAX(created_at) as last_annotation
            FROM annotations
            GROUP BY annotator
            ORDER BY total_annotations DESC
            """
        ).fetchall()
    return pd.DataFrame(rows, columns=[
        "annotator", "total_annotations", "texts_annotated",
        "yes_count", "no_count", "unsure_count",
        "first_annotation", "last_annotation"
    ])


def load_intent_stats() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                c.label,
                i.cluster,
                i.complexity,
                COUNT(DISTINCT c.text_id) as times_in_topk,
                SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END) as yes_count,
                SUM(CASE WHEN a.decision = 'no' THEN 1 ELSE 0 END) as no_count,
                SUM(CASE WHEN a.decision = 'unsure' THEN 1 ELSE 0 END) as unsure_count,
                COUNT(a.id) as total_votes,
                AVG(c.probability) as avg_probability,
                AVG(c.rank) as avg_rank
            FROM candidates c
            LEFT JOIN annotations a ON a.text_id = c.text_id AND a.label = c.label
            LEFT JOIN intents i ON i.label = c.label
            GROUP BY c.label
            ORDER BY times_in_topk DESC
            """
        ).fetchall()
    df = pd.DataFrame(rows, columns=[
        "label", "cluster", "complexity", "times_in_topk",
        "yes_count", "no_count", "unsure_count", "total_votes",
        "avg_probability", "avg_rank"
    ])
    df["yes_count"] = df["yes_count"].fillna(0).astype(int)
    df["no_count"] = df["no_count"].fillna(0).astype(int)
    df["unsure_count"] = df["unsure_count"].fillna(0).astype(int)
    df["total_votes"] = df["total_votes"].fillna(0).astype(int)
    df["precision"] = df.apply(
        lambda row: row["yes_count"] / row["total_votes"] if row["total_votes"] > 0 else None,
        axis=1
    )
    df["unsure_rate"] = df.apply(
        lambda row: row["unsure_count"] / row["total_votes"] if row["total_votes"] > 0 else None,
        axis=1
    )
    return df


def load_disagreements() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                a1.text_id,
                a1.label,
                a1.annotator as annotator1,
                a1.decision as decision1,
                a2.annotator as annotator2,
                a2.decision as decision2,
                t.text
            FROM annotations a1
            JOIN annotations a2 ON a1.text_id = a2.text_id
                AND a1.label = a2.label
                AND a1.annotator < a2.annotator
            JOIN texts t ON t.id = a1.text_id
            WHERE a1.decision != a2.decision
            ORDER BY a1.text_id DESC
            """
        ).fetchall()
    return pd.DataFrame(rows, columns=[
        "text_id", "label", "annotator1", "decision1",
        "annotator2", "decision2", "text"
    ])


def load_annotations_timeline() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                DATE(created_at) as date,
                annotator,
                COUNT(*) as count
            FROM annotations
            GROUP BY DATE(created_at), annotator
            ORDER BY date DESC
            """
        ).fetchall()
    return pd.DataFrame(rows, columns=["date", "annotator", "count"])


def load_cluster_progress() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                t.assigned_cluster,
                COUNT(DISTINCT t.id) as total_texts,
                COUNT(DISTINCT CASE WHEN a.id IS NOT NULL THEN t.id END) as annotated_texts,
                COUNT(DISTINCT a.annotator) as annotators_involved
            FROM texts t
            LEFT JOIN annotations a ON a.text_id = t.id
            GROUP BY t.assigned_cluster
            ORDER BY total_texts DESC
            """
        ).fetchall()
    df = pd.DataFrame(rows, columns=[
        "cluster", "total_texts", "annotated_texts", "annotators_involved"
    ])
    df["progress_pct"] = (df["annotated_texts"] / df["total_texts"] * 100).round(1)
    return df


def load_extra_labels_stats() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                label,
                COUNT(*) as times_added,
                COUNT(DISTINCT text_id) as unique_texts,
                COUNT(DISTINCT annotator) as by_annotators
            FROM annotations
            WHERE is_candidate = 0
            GROUP BY label
            ORDER BY times_added DESC
            """
        ).fetchall()
    return pd.DataFrame(rows, columns=[
        "label", "times_added", "unique_texts", "by_annotators"
    ])


# Overview Section
st.header("Обзор")
stats = load_overview_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего текстов", stats["total_texts"])
col2.metric("Размечено текстов", stats["texts_with_annotations"])
col3.metric("Ожидают разметки", stats["pending_texts"])
col4.metric("Всего аннотаций", stats["total_annotations"])

st.subheader("Тексты по кластерам")
if stats["texts_by_cluster"]:
    cluster_df = pd.DataFrame(
        list(stats["texts_by_cluster"].items()),
        columns=["Кластер", "Количество"]
    )
    st.bar_chart(cluster_df.set_index("Кластер"))

# Cluster Progress
st.header("Прогресс по кластерам")
cluster_progress = load_cluster_progress()
if not cluster_progress.empty:
    st.dataframe(
        cluster_progress.rename(columns={
            "cluster": "Кластер",
            "total_texts": "Всего текстов",
            "annotated_texts": "Размечено",
            "annotators_involved": "Разметчиков",
            "progress_pct": "Прогресс %"
        }),
        use_container_width=True
    )

# Annotator Performance
st.header("Статистика разметчиков")
annotator_stats = load_annotator_stats()
if not annotator_stats.empty:
    st.dataframe(
        annotator_stats.rename(columns={
            "annotator": "Разметчик",
            "total_annotations": "Всего аннотаций",
            "texts_annotated": "Текстов размечено",
            "yes_count": "Yes",
            "no_count": "No",
            "unsure_count": "Unsure",
            "first_annotation": "Первая аннотация",
            "last_annotation": "Последняя аннотация"
        }),
        use_container_width=True
    )

# Annotations Timeline
st.subheader("Активность по дням")
timeline = load_annotations_timeline()
if not timeline.empty:
    pivot = timeline.pivot(index="date", columns="annotator", values="count").fillna(0)
    st.bar_chart(pivot)

# Intent Statistics
st.header("Статистика по интентам")
intent_stats = load_intent_stats()
if not intent_stats.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Фильтры")
        clusters = ["Все"] + sorted(intent_stats["cluster"].dropna().unique().tolist())
        selected_cluster = st.selectbox("Кластер", clusters)
        min_votes = st.slider("Минимум голосов", 0, int(intent_stats["total_votes"].max()) if not intent_stats.empty else 10, 0)

    filtered = intent_stats.copy()
    if selected_cluster != "Все":
        filtered = filtered[filtered["cluster"] == selected_cluster]
    filtered = filtered[filtered["total_votes"] >= min_votes]

    with col2:
        st.subheader("Сортировка")
        sort_options = {
            "По частоте в topK": "times_in_topk",
            "По precision (↑)": "precision",
            "По precision (↓)": "precision",
            "По unsure rate (↑)": "unsure_rate",
            "По unsure rate (↓)": "unsure_rate",
        }
        sort_by = st.selectbox("Сортировать", list(sort_options.keys()))
        ascending = "↑" in sort_by
        filtered = filtered.sort_values(sort_options[sort_by], ascending=ascending, na_position="last")

    st.dataframe(
        filtered.rename(columns={
            "label": "Интент",
            "cluster": "Кластер",
            "complexity": "Сложность",
            "times_in_topk": "В topK",
            "yes_count": "Yes",
            "no_count": "No",
            "unsure_count": "Unsure",
            "total_votes": "Всего голосов",
            "avg_probability": "Ср. вероятность",
            "avg_rank": "Ср. ранг",
            "precision": "Precision",
            "unsure_rate": "Unsure Rate"
        }).style.format({
            "Ср. вероятность": "{:.3f}",
            "Ср. ранг": "{:.1f}",
            "Precision": "{:.2%}",
            "Unsure Rate": "{:.2%}"
        }, na_rep="-"),
        use_container_width=True
    )

    # Worst performing intents
    st.subheader("Проблемные интенты")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Низкий Precision (часто ошибается)**")
        low_precision = intent_stats[intent_stats["total_votes"] >= 3].nsmallest(10, "precision")
        if not low_precision.empty:
            st.dataframe(
                low_precision[["label", "precision", "total_votes", "yes_count", "no_count"]].rename(columns={
                    "label": "Интент",
                    "precision": "Precision",
                    "total_votes": "Голосов",
                    "yes_count": "Yes",
                    "no_count": "No"
                }).style.format({"Precision": "{:.2%}"}, na_rep="-"),
                use_container_width=True
            )

    with col2:
        st.markdown("**Высокий Unsure Rate (сложные для разметки)**")
        high_unsure = intent_stats[intent_stats["total_votes"] >= 3].nlargest(10, "unsure_rate")
        if not high_unsure.empty:
            st.dataframe(
                high_unsure[["label", "unsure_rate", "total_votes", "unsure_count"]].rename(columns={
                    "label": "Интент",
                    "unsure_rate": "Unsure Rate",
                    "total_votes": "Голосов",
                    "unsure_count": "Unsure"
                }).style.format({"Unsure Rate": "{:.2%}"}, na_rep="-"),
                use_container_width=True
            )

# Extra Labels (outside topK)
st.header("Дополнительные метки (вне topK)")
extra_labels = load_extra_labels_stats()
if not extra_labels.empty:
    st.markdown("Интенты, которые разметчики добавляют вручную (модель не предложила в topK):")
    st.dataframe(
        extra_labels.rename(columns={
            "label": "Интент",
            "times_added": "Раз добавлен",
            "unique_texts": "Уникальных текстов",
            "by_annotators": "Разметчиками"
        }),
        use_container_width=True
    )
else:
    st.info("Пока нет дополнительных меток вне topK.")

# Disagreements
st.header("Разногласия между разметчиками")
disagreements = load_disagreements()
if not disagreements.empty:
    st.warning(f"Найдено {len(disagreements)} разногласий")
    st.dataframe(
        disagreements.rename(columns={
            "text_id": "ID текста",
            "label": "Интент",
            "annotator1": "Разметчик 1",
            "decision1": "Решение 1",
            "annotator2": "Разметчик 2",
            "decision2": "Решение 2",
            "text": "Текст"
        }),
        use_container_width=True
    )
else:
    st.success("Разногласий не найдено.")

# Data Export
st.header("Экспорт данных")
col1, col2 = st.columns(2)

with col1:
    if st.button("Экспорт аннотаций в CSV"):
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.id as text_id,
                    t.text,
                    t.assigned_cluster,
                    a.annotator,
                    a.label,
                    a.decision,
                    a.is_candidate,
                    a.created_at
                FROM annotations a
                JOIN texts t ON t.id = a.text_id
                ORDER BY t.id, a.annotator, a.label
                """
            ).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=[
                "text_id", "text", "assigned_cluster", "annotator",
                "label", "decision", "is_candidate", "created_at"
            ])
            csv = df.to_csv(index=False)
            st.download_button(
                "Скачать CSV",
                csv,
                "annotations_export.csv",
                "text/csv"
            )
        else:
            st.info("Нет данных для экспорта.")

with col2:
    if st.button("Экспорт статистики интентов"):
        if not intent_stats.empty:
            csv = intent_stats.to_csv(index=False)
            st.download_button(
                "Скачать CSV",
                csv,
                "intent_stats_export.csv",
                "text/csv"
            )
        else:
            st.info("Нет данных для экспорта.")
