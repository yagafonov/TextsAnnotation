import os
from typing import Dict, List

import pandas as pd
import streamlit as st

from db import connect, create_data_version, create_model_version

ADMIN_PASSWORD = os.environ.get("TEXTS_ADMIN_PASSWORD", "admin123")
MIN_ANNOTATORS = int(os.environ.get("TEXTS_MIN_ANNOTATORS", "2"))

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
            """
            SELECT COUNT(*) as cnt
            FROM texts t
            LEFT JOIN (
                SELECT text_id, COUNT(DISTINCT annotator) as annotators_cnt
                FROM annotations
                GROUP BY text_id
            ) a ON a.text_id = t.id
            WHERE COALESCE(a.annotators_cnt, 0) >= ?
            """,
            (MIN_ANNOTATORS,),
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
                MIN(created_at) as first_annotation,
                MAX(created_at) as last_annotation
            FROM annotations
            GROUP BY annotator
            ORDER BY total_annotations DESC
            """
        ).fetchall()
    return pd.DataFrame(rows, columns=[
        "annotator", "total_annotations", "texts_annotated",
        "yes_count", "no_count",
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
        "yes_count", "no_count", "total_votes",
        "avg_probability", "avg_rank"
    ])
    df["yes_count"] = df["yes_count"].fillna(0).astype(int)
    df["no_count"] = df["no_count"].fillna(0).astype(int)
    df["total_votes"] = df["total_votes"].fillna(0).astype(int)
    df["precision"] = df.apply(
        lambda row: row["yes_count"] / row["total_votes"] if row["total_votes"] > 0 else None,
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


def load_hourly_activity() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                DATE(created_at) as date,
                CAST(strftime('%H', created_at) AS INTEGER) as hour,
                annotator,
                COUNT(*) as count
            FROM annotations
            GROUP BY DATE(created_at), strftime('%H', created_at), annotator
            ORDER BY date DESC, hour
            """
        ).fetchall()
    return pd.DataFrame(rows, columns=["date", "hour", "annotator", "count"])


def load_cluster_progress() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                t.assigned_cluster,
                COUNT(DISTINCT t.id) as total_texts,
                COUNT(DISTINCT CASE WHEN COALESCE(ac.annotators_cnt, 0) >= ? THEN t.id END) as annotated_texts,
                COUNT(DISTINCT a.annotator) as annotators_involved
            FROM texts t
            LEFT JOIN (
                SELECT text_id, COUNT(DISTINCT annotator) as annotators_cnt
                FROM annotations
                GROUP BY text_id
            ) ac ON ac.text_id = t.id
            LEFT JOIN annotations a ON a.text_id = t.id
            GROUP BY t.assigned_cluster
            ORDER BY total_texts DESC
            """,
            (MIN_ANNOTATORS,),
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


def load_intent_model_quality() -> pd.DataFrame:
    """
    Per-intent model quality metrics:
    - top1_precision: when intent is rank 1, how often marked 'yes'
    - top1_count: how many times intent was rank 1
    - missed_opportunity: when intent is rank 2-N and top1 is 'no',
                          how often this intent is marked 'yes'
    """
    with connect() as conn:
        # Get top-1 precision per intent
        top1_stats = conn.execute(
            """
            SELECT
                c.label,
                COUNT(DISTINCT c.text_id || '-' || a.annotator) as top1_count,
                SUM(CASE WHEN a.decision = 'yes' THEN 1 ELSE 0 END) as top1_yes,
                SUM(CASE WHEN a.decision = 'no' THEN 1 ELSE 0 END) as top1_no
            FROM candidates c
            JOIN annotations a ON a.text_id = c.text_id AND a.label = c.label
            WHERE c.rank = 1
            GROUP BY c.label
            """
        ).fetchall()

        # Get missed opportunities: intent is rank 2-N, top1 is no, this intent is yes
        missed_stats = conn.execute(
            """
            SELECT
                c_other.label,
                COUNT(*) as missed_opportunity_count
            FROM candidates c_top1
            JOIN candidates c_other ON c_other.text_id = c_top1.text_id
                AND c_other.rank > 1
            JOIN annotations a_top1 ON a_top1.text_id = c_top1.text_id
                AND a_top1.label = c_top1.label
            JOIN annotations a_other ON a_other.text_id = c_other.text_id
                AND a_other.label = c_other.label
                AND a_other.annotator = a_top1.annotator
            WHERE c_top1.rank = 1
                AND a_top1.decision = 'no'
                AND a_other.decision = 'yes'
            GROUP BY c_other.label
            """
        ).fetchall()

        # Get total times each intent appeared in rank 2-N when top1 was rejected
        potential_missed = conn.execute(
            """
            SELECT
                c_other.label,
                COUNT(DISTINCT c_other.text_id || '-' || a_top1.annotator) as potential_count
            FROM candidates c_top1
            JOIN candidates c_other ON c_other.text_id = c_top1.text_id
                AND c_other.rank > 1
            JOIN annotations a_top1 ON a_top1.text_id = c_top1.text_id
                AND a_top1.label = c_top1.label
            WHERE c_top1.rank = 1
                AND a_top1.decision = 'no'
            GROUP BY c_other.label
            """
        ).fetchall()

        # Get cluster info
        intent_clusters = conn.execute(
            "SELECT label, cluster FROM intents"
        ).fetchall()

    # Build dataframes
    top1_df = pd.DataFrame(top1_stats, columns=[
        "label", "top1_count", "top1_yes", "top1_no"
    ]) if top1_stats else pd.DataFrame(columns=["label", "top1_count", "top1_yes", "top1_no"])

    missed_df = pd.DataFrame(missed_stats, columns=[
        "label", "missed_opportunity_count"
    ]) if missed_stats else pd.DataFrame(columns=["label", "missed_opportunity_count"])

    potential_df = pd.DataFrame(potential_missed, columns=[
        "label", "potential_count"
    ]) if potential_missed else pd.DataFrame(columns=["label", "potential_count"])

    cluster_df = pd.DataFrame(intent_clusters, columns=[
        "label", "cluster"
    ]) if intent_clusters else pd.DataFrame(columns=["label", "cluster"])

    # Merge all
    result = top1_df.merge(missed_df, on="label", how="outer")
    result = result.merge(potential_df, on="label", how="outer")
    result = result.merge(cluster_df, on="label", how="left")

    # Fill NaN with 0
    for col in ["top1_count", "top1_yes", "top1_no", "missed_opportunity_count", "potential_count"]:
        result[col] = result[col].fillna(0).astype(int)

    # Calculate metrics
    result["top1_precision"] = result.apply(
        lambda r: r["top1_yes"] / r["top1_count"] if r["top1_count"] > 0 else None,
        axis=1
    )
    result["missed_rate"] = result.apply(
        lambda r: r["missed_opportunity_count"] / r["potential_count"] if r["potential_count"] > 0 else None,
        axis=1
    )

    return result.sort_values("top1_count", ascending=False)


# Overview Section
st.header("Обзор")
stats = load_overview_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего текстов", stats["total_texts"])
col2.metric(f"Размечено текстов (≥{MIN_ANNOTATORS})", stats["texts_with_annotations"])
col3.metric(f"Ожидают разметки (<{MIN_ANNOTATORS})", stats["pending_texts"])
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

# Hourly Activity
st.subheader("Активность по часам")
hourly = load_hourly_activity()
if not hourly.empty:
    # Date filter
    available_dates = sorted(hourly["date"].unique(), reverse=True)
    selected_date = st.selectbox(
        "Выберите дату",
        available_dates,
        key="hourly_date"
    )

    hourly_filtered = hourly[hourly["date"] == selected_date]
    if not hourly_filtered.empty:
        # Create pivot: hours as index, annotators as columns
        hourly_pivot = hourly_filtered.pivot(index="hour", columns="annotator", values="count").fillna(0)
        # Ensure all hours 0-23 are present
        all_hours = pd.DataFrame(index=range(24))
        hourly_pivot = all_hours.join(hourly_pivot).fillna(0)
        hourly_pivot.index.name = "Час"

        st.bar_chart(hourly_pivot)

        # Summary table for selected date
        st.caption("Сводка по разметчикам за выбранный день:")
        summary = hourly_filtered.groupby("annotator").agg(
            total=("count", "sum"),
            hours_active=("hour", "nunique"),
            first_hour=("hour", "min"),
            last_hour=("hour", "max")
        ).reset_index()
        summary.columns = ["Разметчик", "Всего аннотаций", "Часов активности", "Начало (час)", "Конец (час)"]
        st.dataframe(summary, use_container_width=True, hide_index=True)
else:
    st.info("Нет данных об активности по часам.")

# Intent Model Quality
st.header("Качество модели по интентам")
st.markdown("""
Метрики для понимания, какие интенты модель предсказывает хорошо, а какие — плохо:
- **Top-1 Precision** — когда интент на 1 месте, как часто разметчик ставит "yes"
- **Missed Rate** — когда интент на 2-N месте И top-1 отвергнут (no), как часто этот интент получает "yes"
  (высокий missed rate = модель часто ставит этот интент ниже, чем нужно)
""")

model_quality = load_intent_model_quality()
if not model_quality.empty:
    col1, col2 = st.columns(2)
    with col1:
        mq_clusters = ["Все"] + sorted(model_quality["cluster"].dropna().unique().tolist())
        mq_selected_cluster = st.selectbox("Фильтр по кластеру", mq_clusters, key="mq_cluster")
        mq_min_top1 = st.slider("Минимум Top-1 появлений", 0, int(model_quality["top1_count"].max()) if not model_quality.empty else 10, 0, key="mq_min")

    mq_filtered = model_quality.copy()
    if mq_selected_cluster != "Все":
        mq_filtered = mq_filtered[mq_filtered["cluster"] == mq_selected_cluster]
    mq_filtered = mq_filtered[mq_filtered["top1_count"] >= mq_min_top1]

    with col2:
        mq_sort_options = {
            "По Top-1 частоте (↓)": ("top1_count", False),
            "По Top-1 Precision (↑)": ("top1_precision", True),
            "По Top-1 Precision (↓)": ("top1_precision", False),
            "По Missed Rate (↓)": ("missed_rate", False),
        }
        mq_sort_by = st.selectbox("Сортировка", list(mq_sort_options.keys()), key="mq_sort")
        sort_col, sort_asc = mq_sort_options[mq_sort_by]
        mq_filtered = mq_filtered.sort_values(sort_col, ascending=sort_asc, na_position="last")

    st.dataframe(
        mq_filtered[["label", "cluster", "top1_count", "top1_yes", "top1_no",
                     "top1_precision", "potential_count",
                     "missed_opportunity_count", "missed_rate"]].rename(columns={
            "label": "Интент",
            "cluster": "Кластер",
            "top1_count": "Top-1 раз",
            "top1_yes": "Top-1 Yes",
            "top1_no": "Top-1 No",
            "top1_precision": "Top-1 Precision",
            "potential_count": "Потенц. пропусков",
            "missed_opportunity_count": "Факт. пропусков",
            "missed_rate": "Missed Rate"
        }).style.format({
            "Top-1 Precision": "{:.1%}",
            "Missed Rate": "{:.1%}"
        }, na_rep="-"),
        use_container_width=True
    )

    # Highlight problematic intents
    st.subheader("Интенты, требующие внимания")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Низкий Top-1 Precision**")
        st.caption("Модель уверена, но ошибается")
        low_top1_prec = model_quality[model_quality["top1_count"] >= 3].nsmallest(10, "top1_precision")
        if not low_top1_prec.empty:
            st.dataframe(
                low_top1_prec[["label", "top1_precision", "top1_count", "top1_no"]].rename(columns={
                    "label": "Интент",
                    "top1_precision": "Precision",
                    "top1_count": "Top-1 раз",
                    "top1_no": "No"
                }).style.format({"Precision": "{:.1%}"}, na_rep="-"),
                use_container_width=True,
                hide_index=True
            )

    with col2:
        st.markdown("**Высокий Missed Rate**")
        st.caption("Модель недооценивает этот интент")
        high_missed = model_quality[model_quality["potential_count"] >= 3].nlargest(10, "missed_rate")
        if not high_missed.empty:
            st.dataframe(
                high_missed[["label", "missed_rate", "missed_opportunity_count", "potential_count"]].rename(columns={
                    "label": "Интент",
                    "missed_rate": "Missed Rate",
                    "missed_opportunity_count": "Пропущено",
                    "potential_count": "Потенциал"
                }).style.format({"Missed Rate": "{:.1%}"}, na_rep="-"),
                use_container_width=True,
                hide_index=True
            )
else:
    st.info("Недостаточно данных для расчета метрик качества модели.")

st.divider()

# Intent Statistics
st.header("Статистика по интентам (общая)")
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
            "total_votes": "Всего голосов",
            "avg_probability": "Ср. вероятность",
            "avg_rank": "Ср. ранг",
            "precision": "Precision"
        }).style.format({
            "Ср. вероятность": "{:.3f}",
            "Ср. ранг": "{:.1f}",
            "Precision": "{:.2%}"
        }, na_rep="-"),
        use_container_width=True
    )

    # Worst performing intents
    st.subheader("Проблемные интенты")
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

# Model Training Stub (Admin only)
st.header("Управление версиями модели")
note = st.text_input("Комментарий к версии", value="training stub", key="admin_training_note")
if st.button("Создать новую версию модели"):
    with connect() as conn:
        new_model_version = create_model_version(conn, note)
        new_data_version = create_data_version(conn)
    st.info(f"Создана модель версии {new_model_version}. Новая версия данных: {new_data_version}.")

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
