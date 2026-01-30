import os
from datetime import datetime
from typing import Dict, List

import streamlit as st
import yaml

from db import (
    DEFAULT_DB_PATH,
    DEFAULT_DUMP_INTERVAL_SEC,
    DEFAULT_DUMP_PATH,
    connect,
    create_data_version,
    create_model_version,
    get_setting,
    init_db,
    restore_from_dump_if_needed,
    start_auto_dump,
)
from modeling import Candidate, TopKModelStub, compute_metrics

INTENTS_PATH = os.environ.get("TEXTS_INTENTS_PATH", "data/intents.yaml")
MARGIN_THRESHOLD = float(os.environ.get("TEXTS_MARGIN_THRESHOLD", "0.1"))


def _load_yaml_file(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@st.cache_resource
def load_intents(path: str) -> Dict[str, dict]:
    if os.path.isdir(path):
        intents: Dict[str, dict] = {}
        for filename in sorted(os.listdir(path)):
            if not filename.lower().endswith((".yaml", ".yml")):
                continue
            file_path = os.path.join(path, filename)
            intents.update(_load_yaml_file(file_path))
        return intents
    return _load_yaml_file(path)


@st.cache_resource
def init_services():
    restore_from_dump_if_needed(DEFAULT_DB_PATH, DEFAULT_DUMP_PATH)
    init_db(DEFAULT_DB_PATH)
    start_auto_dump(DEFAULT_DB_PATH, DEFAULT_DUMP_PATH, DEFAULT_DUMP_INTERVAL_SEC)
    intents = load_intents(INTENTS_PATH)
    model = TopKModelStub(intents)
    return intents, model


def add_text(
    text: str,
    language: str,
    clusters: str,
    candidates: List[Candidate],
    model_version: int,
    data_version: int,
) -> int:
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO texts (text, language, clusters, data_version, created_at) VALUES (?, ?, ?, ?, ?)",
            (text, language, clusters, data_version, datetime.utcnow().isoformat()),
        )
        text_id = cursor.lastrowid
        for candidate in candidates:
            conn.execute(
                """
                INSERT INTO candidates (text_id, label, rank, probability, model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    text_id,
                    candidate.label,
                    candidate.rank,
                    candidate.probability,
                    model_version,
                    datetime.utcnow().isoformat(),
                ),
            )
        conn.commit()
        return text_id


def save_annotations(
    text_id: int,
    annotator: str,
    decisions: Dict[str, str],
    candidate_labels: List[str],
    extra_labels: List[str],
) -> None:
    with connect() as conn:
        for label, decision in decisions.items():
            conn.execute(
                """
                INSERT INTO annotations (text_id, annotator, label, decision, is_candidate, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    text_id,
                    annotator,
                    label,
                    decision,
                    1 if label in candidate_labels else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        for label in extra_labels:
            conn.execute(
                """
                INSERT INTO annotations (text_id, annotator, label, decision, is_candidate, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    text_id,
                    annotator,
                    label,
                    "yes",
                    0,
                    datetime.utcnow().isoformat(),
                ),
            )
        conn.commit()


st.set_page_config(page_title="Texts Annotation", layout="wide")

intents, model = init_services()

st.title("Платформа разметки текстов")

with st.sidebar:
    st.header("Настройки разметчика")
    annotator = st.text_input("Имя разметчика", value=st.session_state.get("annotator", ""))
    if annotator:
        st.session_state["annotator"] = annotator

    st.divider()
    st.header("Импорт новых текстов")
    new_text = st.text_area("Текст для разметки", height=120)
    language = st.text_input("Язык фразы", value="ru")
    clusters = st.text_input("Кластеры (через запятую)")
    if st.button("Добавить текст") and new_text:
        with connect() as conn:
            model_version = int(get_setting(conn, "current_model_version", "0"))
            data_version = int(get_setting(conn, "current_data_version", "0"))
        candidates = model.predict(new_text)
        text_id = add_text(new_text, language, clusters, candidates, model_version, data_version)
        st.success(f"Текст добавлен (ID {text_id}).")

    st.divider()
    st.header("Обучение модели (заглушка)")
    note = st.text_input("Комментарий к версии", value="training stub")
    if st.button("Запустить обучение"):
        with connect() as conn:
            new_model_version = create_model_version(conn, note)
            new_data_version = create_data_version(conn)
        st.info(f"Создана модель версии {new_model_version}. Новая версия данных: {new_data_version}.")

st.subheader("Список текстов")
with connect() as conn:
    texts = conn.execute(
        """
        SELECT t.id, t.text, t.language, t.clusters, t.data_version, t.created_at,
               COUNT(DISTINCT a.annotator) as annotators
        FROM texts t
        LEFT JOIN annotations a ON a.text_id = t.id
        GROUP BY t.id
        ORDER BY t.created_at DESC
        """
    ).fetchall()

if not texts:
    st.info("Добавьте тексты для разметки.")
    st.stop()

text_options = {f"#{row['id']} ({row['annotators']} разметчика)": row["id"] for row in texts}
selected_label = st.selectbox("Выберите текст для разметки", list(text_options.keys()))
selected_text_id = text_options[selected_label]

with connect() as conn:
    text_row = conn.execute("SELECT * FROM texts WHERE id = ?", (selected_text_id,)).fetchone()
    candidates_rows = conn.execute(
        "SELECT * FROM candidates WHERE text_id = ? ORDER BY rank ASC",
        (selected_text_id,),
    ).fetchall()
    existing_annotations = conn.execute(
        "SELECT * FROM annotations WHERE text_id = ?",
        (selected_text_id,),
    ).fetchall()

st.markdown("### Текст")
st.write(text_row["text"])

st.caption(
    f"Язык: {text_row['language']} | Кластеры: {text_row['clusters'] or '-'} | Версия данных: {text_row['data_version']}"
)

candidate_labels = [row["label"] for row in candidates_rows]

st.markdown("### Кандидаты (topK)")

col1, col2 = st.columns([2, 1])
with col1:
    decisions: Dict[str, str] = {}
    for row in candidates_rows:
        label = row["label"]
        label_info = intents.get(label, {})
        decision = st.radio(
            f"{label}",
            ["yes", "no", "unsure"],
            horizontal=True,
            key=f"decision_{selected_text_id}_{label}",
        )
        decisions[label] = decision
        with st.expander("Описание интента"):
            st.write(label_info.get("description", "Нет описания."))
            st.write("Примеры:")
            st.write(", ".join(label_info.get("train", [])[:5]) or "Нет примеров"
            )

with col2:
    st.markdown("### Метаданные")
    for row in candidates_rows:
        label = row["label"]
        label_info = intents.get(label, {})
        st.write(
            f"**{label}** | rank={row['rank']} | p={row['probability']:.2f} | "
            f"сложность={label_info.get('complexity', '-')}, кластер={label_info.get('cluster', '-') }"
        )

all_intents = list(intents.keys())
extra_labels = st.multiselect(
    "Дополнительные метки вне topK",
    [label for label in all_intents if label not in candidate_labels],
)

if st.button("Сохранить разметку"):
    if not annotator:
        st.error("Укажите имя разметчика в боковой панели.")
    else:
        save_annotations(selected_text_id, annotator, decisions, candidate_labels, extra_labels)
        st.success("Разметка сохранена.")

st.markdown("### Статистика")
with connect() as conn:
    candidates_rows = conn.execute(
        """
        SELECT text_id, label, rank, probability
        FROM candidates
        ORDER BY text_id, rank
        """
    ).fetchall()
    annotations_rows = conn.execute(
        """
        SELECT text_id, annotator, label, decision
        FROM annotations
        WHERE decision = 'yes'
        """
    ).fetchall()

if candidates_rows and annotations_rows:
    candidates_by_text = {}
    for row in candidates_rows:
        candidates_by_text.setdefault(row["text_id"], []).append(
            Candidate(label=row["label"], rank=row["rank"], probability=row["probability"])
        )

    metrics_by_text = {}
    for row in annotations_rows:
        key = (row["text_id"], row["annotator"])
        metrics_by_text.setdefault(key, {"candidates": candidates_by_text.get(row["text_id"], []), "targets": []})
        metrics_by_text[key]["targets"].append(row["label"])

    totals = {"top1_hit_rate": 0, "topK_coverage": 0, "margin_error_rate": 0, "outside_topK_rate": 0}
    count = 0
    for entry in metrics_by_text.values():
        metrics = compute_metrics(entry["targets"], entry["candidates"], MARGIN_THRESHOLD)
        for key in totals:
            totals[key] += 1 if metrics[key] else 0
        count += 1

    if count:
        st.write({key: round(value / count, 3) for key, value in totals.items()})
else:
    st.info("Недостаточно данных для расчета метрик.")

st.markdown("### История разметок")
if existing_annotations:
    st.dataframe(existing_annotations, use_container_width=True)
else:
    st.info("Для этого текста еще нет разметок.")
