import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple

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
    upsert_intent,
)
from modeling import Candidate, TopKModelStub, compute_metrics

INTENTS_PATH = os.environ.get("TEXTS_INTENTS_PATH", "data/intents")
ANNOTATORS_PATH = os.environ.get("TEXTS_ANNOTATORS_PATH", "data/annotators.yaml")
IMPORT_CSV_PATH = os.environ.get("TEXTS_IMPORT_CSV_PATH", "data/requests.csv")
MARGIN_THRESHOLD = float(os.environ.get("TEXTS_MARGIN_THRESHOLD", "0.1"))


def _load_yaml_file(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@st.cache_resource
def load_intents(path: str) -> Tuple[Dict[str, dict], Dict[str, str]]:
    intents: Dict[str, dict] = {}
    intent_sources: Dict[str, str] = {}
    if os.path.isdir(path):
        for filename in sorted(os.listdir(path)):
            if not filename.lower().endswith((".yaml", ".yml")):
                continue
            file_path = os.path.join(path, filename)
            cluster_name = os.path.splitext(filename)[0]
            file_intents = _load_yaml_file(file_path)
            for label, payload in file_intents.items():
                payload = payload or {}
                payload.setdefault("cluster", cluster_name)
                intents[label] = payload
                intent_sources[label] = filename
        return intents, intent_sources
    single_intents = _load_yaml_file(path)
    for label, payload in single_intents.items():
        intents[label] = payload or {}
        intent_sources[label] = os.path.basename(path)
    return intents, intent_sources


def load_annotators(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    data = _load_yaml_file(path)
    if isinstance(data, dict) and "annotators" in data:
        return data["annotators"] or []
    if isinstance(data, list):
        return data
    return []


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    divider = 10.0 if max_score > 1 else 1.0
    return {label: value / divider for label, value in scores.items()}


def select_top_k(scores: Dict[str, float], top_k: int) -> List[Candidate]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [Candidate(label=label, rank=idx + 1, probability=score) for idx, (label, score) in enumerate(ranked)]


def determine_cluster(candidates: List[Candidate], intents: Dict[str, dict]) -> str:
    cluster_counts: Dict[str, float] = {}
    for candidate in candidates:
        cluster = intents.get(candidate.label, {}).get("cluster")
        if not cluster:
            continue
        cluster_counts[cluster] = cluster_counts.get(cluster, 0.0) + candidate.probability
    if not cluster_counts:
        return "unknown"
    return max(cluster_counts.items(), key=lambda item: item[1])[0]


def import_texts_from_csv(
    path: str,
    intents: Dict[str, dict],
    top_k: int,
) -> int:
    if not os.path.exists(path):
        return 0
    added = 0
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "request_text" not in reader.fieldnames:
            return 0
        for row in reader:
            request_text = (row.get("request_text") or "").strip()
            if not request_text:
                continue
            raw_scores: Dict[str, float] = {}
            for key, value in row.items():
                if key == "request_text" or value is None:
                    continue
                if key not in intents:
                    continue
                try:
                    raw_scores[key] = float(value)
                except ValueError:
                    continue
            scores = normalize_scores(raw_scores)
            candidates = select_top_k(scores, top_k)
            if not candidates:
                continue
            assigned_cluster = determine_cluster(candidates, intents)
            with connect() as conn:
                existing = conn.execute("SELECT id FROM texts WHERE text = ?", (request_text,)).fetchone()
                if existing:
                    continue
                model_version = int(get_setting(conn, "current_model_version", "0"))
                data_version = int(get_setting(conn, "current_data_version", "0"))
                cursor = conn.execute(
                    """
                    INSERT INTO texts (text, language, clusters, assigned_cluster, data_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (request_text, None, None, assigned_cluster, data_version, datetime.utcnow().isoformat()),
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
                added += 1
    return added


@st.cache_resource
def init_services():
    restore_from_dump_if_needed(DEFAULT_DB_PATH, DEFAULT_DUMP_PATH)
    init_db(DEFAULT_DB_PATH)
    start_auto_dump(DEFAULT_DB_PATH, DEFAULT_DUMP_PATH, DEFAULT_DUMP_INTERVAL_SEC)
    intents, intent_sources = load_intents(INTENTS_PATH)
    with connect() as conn:
        for label, payload in intents.items():
            upsert_intent(
                conn,
                label=label,
                description=payload.get("description", ""),
                examples=", ".join(payload.get("train", []) or []),
                complexity=str(payload.get("complexity", "")),
                cluster=str(payload.get("cluster", "")),
                source_file=intent_sources.get(label, ""),
            )
    model = TopKModelStub(intents)
    import_texts_from_csv(IMPORT_CSV_PATH, intents, model.top_k)
    annotators = load_annotators(ANNOTATORS_PATH)
    return intents, model, annotators


def add_text(
    text: str,
    language: str,
    clusters: str,
    candidates: List[Candidate],
    model_version: int,
    data_version: int,
    assigned_cluster: str | None = None,
) -> int:
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO texts (text, language, clusters, assigned_cluster, data_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (text, language, clusters, assigned_cluster, data_version, datetime.utcnow().isoformat()),
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

intents, model, annotators = init_services()

st.title("Платформа разметки текстов")

with st.sidebar:
    st.header("Вход разметчика")
    annotator_names = [item.get("name") for item in annotators]
    annotator_names = [name for name in annotator_names if name]
    annotator = st.selectbox("Разметчик", annotator_names) if annotator_names else st.text_input("Имя разметчика")
    password = st.text_input("Пароль", type="password")
    selected_annotator = next((item for item in annotators if item.get("name") == annotator), {})
    annotator_cluster = selected_annotator.get("cluster")
    password_ok = bool(selected_annotator) and password == selected_annotator.get("password")
    if annotators and not password_ok:
        st.warning("Введите корректный пароль для выбранного разметчика.")

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
        assigned_cluster = determine_cluster(candidates, intents)
        text_id = add_text(
            new_text,
            language,
            clusters,
            candidates,
            model_version,
            data_version,
            assigned_cluster,
        )
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
if annotators and not password_ok:
    st.stop()

show_all = st.checkbox("Показать все тексты (без фильтра по кластеру)")
with connect() as conn:
    base_query = """
        SELECT t.id, t.text, t.language, t.clusters, t.assigned_cluster, t.data_version, t.created_at,
               COUNT(DISTINCT a.annotator) as annotators
        FROM texts t
        LEFT JOIN annotations a ON a.text_id = t.id
    """
    filters = []
    params: List[str] = []
    if annotator:
        filters.append("NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.annotator = ?)")
        params.append(annotator)
    if annotator_cluster and not show_all:
        filters.append("t.assigned_cluster = ?")
        params.append(annotator_cluster)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    texts = conn.execute(
        f\"\"\"\n{base_query}\n{where_clause}\nGROUP BY t.id\nORDER BY t.created_at DESC\n\"\"\",\n        params,\n    ).fetchall()

if not texts:
    st.info("Добавьте тексты для разметки.")
    st.stop()

text_options = {
    f"#{row['id']} [{row['assigned_cluster'] or 'unknown'}] ({row['annotators']} разметчика)": row["id"]
    for row in texts
}
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
    " | ".join(
        [
            f"Язык: {text_row['language']}",
            f"Кластеры: {text_row['clusters'] or '-'}",
            f"Назначенный кластер: {text_row['assigned_cluster'] or '-'}",
            f"Версия данных: {text_row['data_version']}",
        ]
    )
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
