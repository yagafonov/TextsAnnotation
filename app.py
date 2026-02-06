import csv
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import streamlit as st
import streamlit.components.v1 as components
import yaml

from db import (
    DEFAULT_DB_PATH,
    DEFAULT_DUMP_INTERVAL_SEC,
    DEFAULT_DUMP_PATH,
    connect,
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
MIN_ANNOTATORS = int(os.environ.get("TEXTS_MIN_ANNOTATORS", "2"))


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


def normalize_clusters(annotator: dict) -> List[str]:
    clusters = annotator.get("clusters")
    if isinstance(clusters, str):
        return [item.strip() for item in clusters.split(",") if item.strip()]
    if isinstance(clusters, list):
        return [str(item).strip() for item in clusters if str(item).strip()]
    cluster = annotator.get("cluster")
    if isinstance(cluster, list):
        return [str(item).strip() for item in cluster if str(item).strip()]
    if cluster:
        return [str(cluster).strip()]
    return []


def get_annotator_language(annotator: dict) -> str | None:
    """Get annotator's language (handles typo 'languge' in config)."""
    lang = annotator.get("language") or annotator.get("languge")
    if lang:
        return str(lang).strip().lower()
    return None


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    divider = 10.0 if max_score > 1 else 1.0
    return {label: value / divider for label, value in scores.items()}


def select_top_k(scores: Dict[str, float], top_k: int) -> List[Candidate]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [Candidate(label=label, rank=idx + 1, probability=score) for idx, (label, score) in enumerate(ranked)]


def determine_cluster(scores: Dict[str, float], intents: Dict[str, dict]) -> str:
    """Determine cluster by summing scores for all intents belonging to each cluster."""
    cluster_scores: Dict[str, float] = {}
    for label, score in scores.items():
        cluster = intents.get(label, {}).get("cluster")
        if not cluster:
            continue
        cluster_scores[cluster] = cluster_scores.get(cluster, 0.0) + score
    if not cluster_scores:
        return "unknown"
    return max(cluster_scores.items(), key=lambda item: item[1])[0]


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
            # Read language column if present (kz or ru)
            language = (row.get("language") or "").strip().lower() or None
            raw_scores: Dict[str, float] = {}
            for key, value in row.items():
                if key in ("request_text", "language") or value is None:
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
            assigned_cluster = determine_cluster(scores, intents)
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
                    (request_text, language, None, assigned_cluster, data_version, datetime.now(UTC).isoformat()),
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
                            datetime.now(UTC).isoformat(),
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
            (text, language, clusters, assigned_cluster, data_version, datetime.now(UTC).isoformat()),
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
                    datetime.now(UTC).isoformat(),
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
    shown_intents_source: Dict[str, str],
) -> None:
    with connect() as conn:
        now = datetime.now(UTC).isoformat()
        # Save annotations for all shown intents
        for label, decision in decisions.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO annotations (text_id, annotator, label, decision, is_candidate, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    text_id,
                    annotator,
                    label,
                    decision,
                    1 if label in candidate_labels else 0,
                    now,
                ),
            )
        # Save extra labels (intents outside union, selected by annotator)
        for label in extra_labels:
            conn.execute(
                """
                INSERT OR REPLACE INTO annotations (text_id, annotator, label, decision, is_candidate, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    text_id,
                    annotator,
                    label,
                    "yes",
                    0,
                    now,
                ),
            )
        # Save shown_intents for this annotator (union + extra)
        for label, source in shown_intents_source.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO shown_intents (text_id, annotator, label, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (text_id, annotator, label, source, now),
            )
        # Save extra labels as shown_intents with source "extra"
        for label in extra_labels:
            conn.execute(
                """
                INSERT OR REPLACE INTO shown_intents (text_id, annotator, label, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (text_id, annotator, label, "extra", now),
            )
        conn.commit()


st.set_page_config(page_title="Texts Annotation", layout="wide")

# Scroll to top after save/skip
if st.session_state.get("scroll_to_top"):
    st.session_state.scroll_to_top = False
    components.html(
        "<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>",
        height=0,
    )

intents, model, annotators = init_services()

# Build annotator lookup for validation
annotator_lookup = {item.get("name"): item for item in annotators if item.get("name")}

# Initialize session state from URL query params (persists across refresh)
query_params = st.query_params
url_user = query_params.get("user")
url_cluster = query_params.get("cluster")

# Restore session from URL if valid user
if url_user and url_user in annotator_lookup:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
        st.session_state.annotator_name = url_user
        st.session_state.annotator_cluster = url_cluster
else:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "annotator_name" not in st.session_state:
        st.session_state.annotator_name = None
    if "annotator_cluster" not in st.session_state:
        st.session_state.annotator_cluster = None

st.title("Платформа разметки текстов")

with st.sidebar:
    st.header("Вход разметчика")

    # If already logged in, show status and logout button
    if st.session_state.logged_in and st.session_state.annotator_name:
        st.success(f"Вы вошли как: **{st.session_state.annotator_name}**")

        # Get annotator info
        selected_annotator = annotator_lookup.get(st.session_state.annotator_name, {})
        annotator_clusters = normalize_clusters(selected_annotator)
        annotator_language = get_annotator_language(selected_annotator)

        # Cluster selector (persisted in session state and URL)
        annotator_cluster = None
        if annotator_clusters:
            # Use stored cluster or default to first
            default_idx = 0
            if st.session_state.annotator_cluster in annotator_clusters:
                default_idx = annotator_clusters.index(st.session_state.annotator_cluster)
            annotator_cluster = st.selectbox(
                "Активный кластер",
                annotator_clusters,
                index=default_idx,
                key="cluster_selector"
            )
            st.session_state.annotator_cluster = annotator_cluster
            # Update URL with cluster
            st.query_params["cluster"] = annotator_cluster

        if annotator_language:
            st.info(f"Язык: {annotator_language.upper()}")

        # Use session state values
        annotator = st.session_state.annotator_name
        password_ok = True

        # Logout button
        if st.button("Выйти", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.annotator_name = None
            st.session_state.annotator_cluster = None
            # Clear URL params
            st.query_params.clear()
            st.rerun()

    else:
        # Login form
        annotator_names = list(annotator_lookup.keys())

        annotator_input = st.selectbox(
            "Разметчик", annotator_names, key="login_annotator"
        ) if annotator_names else st.text_input("Имя разметчика", key="login_annotator_text")

        password_input = st.text_input("Пароль", type="password", key="login_password")

        selected_annotator = annotator_lookup.get(annotator_input, {})

        # Login button
        if st.button("Войти", type="primary"):
            if selected_annotator and password_input == selected_annotator.get("password"):
                st.session_state.logged_in = True
                st.session_state.annotator_name = annotator_input
                # Save to URL for persistence across refresh
                st.query_params["user"] = annotator_input
                st.toast(f"Добро пожаловать, {annotator_input}!")
                st.rerun()
            else:
                st.error("Неверный пароль.")

        # Not logged in yet
        annotator = None
        password_ok = False
        annotator_clusters = []
        annotator_language = None
        annotator_cluster = None

    # Progress counter for annotator (only if logged in)
    if st.session_state.logged_in and st.session_state.annotator_name:
        with connect() as conn:
            # Get total texts available for this annotator's clusters/language
            total_query_parts = ["SELECT COUNT(*) as cnt FROM texts t WHERE 1=1"]
            total_params = []
            if annotator_cluster:
                total_query_parts.append("AND t.assigned_cluster = ?")
                total_params.append(annotator_cluster)
            elif annotator_clusters:
                placeholders = ", ".join("?" for _ in annotator_clusters)
                total_query_parts.append(f"AND t.assigned_cluster IN ({placeholders})")
                total_params.extend(annotator_clusters)
            if annotator_language:
                total_query_parts.append("AND (t.language = ? OR t.language IS NULL)")
                total_params.append(annotator_language)

            total_texts = conn.execute(" ".join(total_query_parts), total_params).fetchone()["cnt"]

            # Get texts annotated by this annotator
            done_query_parts = [
                """
                SELECT COUNT(DISTINCT t.id) as cnt
                FROM texts t
                INNER JOIN annotations a ON a.text_id = t.id AND a.annotator = ?
                WHERE 1=1
                """
            ]
            done_params = [annotator]
            if annotator_cluster:
                done_query_parts.append("AND t.assigned_cluster = ?")
                done_params.append(annotator_cluster)
            elif annotator_clusters:
                placeholders = ", ".join("?" for _ in annotator_clusters)
                done_query_parts.append(f"AND t.assigned_cluster IN ({placeholders})")
                done_params.extend(annotator_clusters)
            if annotator_language:
                done_query_parts.append("AND (t.language = ? OR t.language IS NULL)")
                done_params.append(annotator_language)

            done_texts = conn.execute(" ".join(done_query_parts), done_params).fetchone()["cnt"]

        st.divider()
        st.subheader("Прогресс")
        progress_pct = done_texts / total_texts if total_texts > 0 else 0
        st.progress(progress_pct)
        st.metric("Размечено", f"{done_texts} / {total_texts}", delta=f"{progress_pct:.1%}")

st.subheader("Список текстов")
if annotators and not password_ok:
    st.stop()

show_skipped = st.checkbox("Показать пропущенные тексты")

with connect() as conn:
    if show_skipped and annotator:
        # Show only skipped texts for this annotator
        base_query = """
            SELECT
                t.id,
                t.text,
                t.language,
                t.clusters,
                t.assigned_cluster,
                t.data_version,
                t.created_at,
                COUNT(DISTINCT a.annotator) as annotators,
                1 as is_skipped
            FROM texts t
            LEFT JOIN annotations a ON a.text_id = t.id
            INNER JOIN skipped_texts s ON s.text_id = t.id AND s.annotator = ?
        """
        filters = []
        params: List[str] = [annotator]
        # Exclude already annotated by this user
        filters.append("NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.annotator = ?)")
        params.append(annotator)
        if annotator_cluster:
            # Filter by selected cluster
            filters.append("t.assigned_cluster = ?")
            params.append(annotator_cluster)
        elif annotator_clusters:
            # Filter by all annotator's allowed clusters
            placeholders = ", ".join("?" for _ in annotator_clusters)
            filters.append(f"t.assigned_cluster IN ({placeholders})")
            params.extend(annotator_clusters)
        # Filter by annotator's language
        if annotator_language:
            filters.append("(t.language = ? OR t.language IS NULL)")
            params.append(annotator_language)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"{base_query} {where_clause} GROUP BY t.id HAVING COUNT(DISTINCT a.annotator) < ? ORDER BY t.assigned_cluster, t.created_at DESC"
        texts = conn.execute(query, params + [MIN_ANNOTATORS]).fetchall()
    else:
        # Normal mode - show unannotated and unskipped texts
        base_query = """
            SELECT
                t.id,
                t.text,
                t.language,
                t.clusters,
                t.assigned_cluster,
                t.data_version,
                t.created_at,
                COUNT(DISTINCT a.annotator) as annotators
            FROM texts t
            LEFT JOIN annotations a ON a.text_id = t.id
        """
        filters = []
        params: List[str] = []
        if annotator:
            filters.append("NOT EXISTS (SELECT 1 FROM annotations a2 WHERE a2.text_id = t.id AND a2.annotator = ?)")
            params.append(annotator)
            filters.append("NOT EXISTS (SELECT 1 FROM skipped_texts s WHERE s.text_id = t.id AND s.annotator = ?)")
            params.append(annotator)
        if annotator_cluster:
            # Filter by selected cluster
            filters.append("t.assigned_cluster = ?")
            params.append(annotator_cluster)
        elif annotator_clusters:
            # Filter by all annotator's allowed clusters
            placeholders = ", ".join("?" for _ in annotator_clusters)
            filters.append(f"t.assigned_cluster IN ({placeholders})")
            params.extend(annotator_clusters)
        # Filter by annotator's language
        if annotator_language:
            filters.append("(t.language = ? OR t.language IS NULL)")
            params.append(annotator_language)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"{base_query} {where_clause} GROUP BY t.id HAVING COUNT(DISTINCT a.annotator) < ? ORDER BY t.assigned_cluster, t.created_at DESC"
        texts = conn.execute(query, params + [MIN_ANNOTATORS]).fetchall()

if not texts:
    if show_skipped:
        st.info("Нет пропущенных текстов.")
    else:
        st.info("Нет текстов для разметки.")
    st.stop()

# Group texts by cluster for display
cluster_counts: Dict[str, int] = {}
for row in texts:
    cluster = row["assigned_cluster"] or "unknown"
    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

# Show cluster summary
if not annotator_cluster and len(cluster_counts) > 1:
    st.caption("Тексты по кластерам: " + " | ".join(f"{c}: {n}" for c, n in sorted(cluster_counts.items())))

if show_skipped:
    text_options = {
        f"[{row['assigned_cluster'] or 'unknown'}] #{row['id']} ({row['annotators']} разметчика) [ПРОПУЩЕН]": row["id"]
        for row in texts
    }
    st.caption(f"Пропущенных текстов: {len(texts)}")
else:
    text_options = {
        f"[{row['assigned_cluster'] or 'unknown'}] #{row['id']} ({row['annotators']} разметчика)": row["id"]
        for row in texts
    }
text_labels = list(text_options.keys())
text_ids = list(text_options.values())
if "selected_text_id" not in st.session_state or st.session_state.selected_text_id not in text_ids:
    st.session_state.selected_text_id = text_ids[0]
default_index = text_ids.index(st.session_state.selected_text_id)
selected_label = st.selectbox("Выберите текст для разметки", text_labels, index=default_index)
selected_text_id = text_options[selected_label]
if selected_text_id != st.session_state.selected_text_id:
    st.session_state.selected_text_id = selected_text_id


def set_next_skipped_text_id() -> None:
    if not show_skipped:
        return
    if selected_text_id not in text_ids:
        return
    current_index = text_ids.index(selected_text_id)
    next_id = None
    if current_index + 1 < len(text_ids):
        next_id = text_ids[current_index + 1]
    elif current_index > 0:
        next_id = text_ids[current_index - 1]
    st.session_state.selected_text_id = next_id

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

# Build candidate lookup for probability/rank info
candidate_lookup = {row["label"]: row for row in candidates_rows}
candidate_labels = [row["label"] for row in candidates_rows]

# Build union of TopK candidates + all intents from annotator's cluster
if annotator_cluster:
    cluster_intent_labels = [
        label for label, payload in intents.items()
        if payload.get("cluster") == annotator_cluster
    ]
else:
    cluster_intent_labels = []

# Union: TopK candidates + cluster intents (deduplicated, preserving order)
shown_intent_labels = list(candidate_labels)  # Start with TopK
for label in cluster_intent_labels:
    if label not in shown_intent_labels:
        shown_intent_labels.append(label)

# Track source for each shown intent
shown_intents_source: Dict[str, str] = {}
for label in shown_intent_labels:
    if label in candidate_labels and label in cluster_intent_labels:
        shown_intents_source[label] = "topk_and_cluster"
    elif label in candidate_labels:
        shown_intents_source[label] = "topk"
    else:
        shown_intents_source[label] = "cluster"

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

st.markdown("### Интенты для разметки (TopK + кластер)")

decisions: Dict[str, str] = {}
for label in shown_intent_labels:
    label_info = intents.get(label, {})
    candidate_row = candidate_lookup.get(label)

    # Create a container for each intent
    with st.container():
        col_check, col_info = st.columns([1, 3])

        with col_check:
            is_yes = st.checkbox(
                f"{label}",
                value=False,
                key=f"decision_{selected_text_id}_{label}",
            )
            decisions[label] = "yes" if is_yes else "no"
            # Show source and probability/rank if available
            source = shown_intents_source[label]
            if candidate_row:
                st.caption(f"[{source}] rank={candidate_row['rank']} | p={candidate_row['probability']:.2f}")
            else:
                st.caption(f"[{source}]")

        with col_info:
            # Description always visible
            description = label_info.get("description", "Нет описания.")
            st.markdown(f"**Описание:** {description}")

            # Examples always visible
            examples = label_info.get("train", [])[:5]
            if examples:
                examples_text = " | ".join(examples)
                st.caption(f"Примеры: {examples_text}")

        st.divider()

# Extra intents: all intents NOT in the shown union (from other clusters)
available_extra_intents = [label for label in intents.keys() if label not in shown_intent_labels]

# Group by cluster for better navigation
all_clusters = sorted(set(payload.get("cluster", "unknown") for payload in intents.values()))
extra_options = []
for cluster in all_clusters:
    cluster_intents = [
        label for label in available_extra_intents
        if intents.get(label, {}).get("cluster") == cluster
    ]
    extra_options.extend(cluster_intents)

extra_labels = st.multiselect(
    "Дополнительные метки вне объединения (другие кластеры)",
    extra_options,
    format_func=lambda x: f"[{intents.get(x, {}).get('cluster', 'unknown')}] {x}",
    key=f"extra_labels_{selected_text_id}",
)

col_save, col_skip = st.columns([1, 1])
with col_save:
    if st.button("Сохранить разметку", type="primary"):
        if not annotator:
            st.error("Укажите имя разметчика в боковой панели.")
        else:
            save_annotations(selected_text_id, annotator, decisions, candidate_labels, extra_labels, shown_intents_source)
            # Remove from skipped if it was there
            with connect() as conn:
                conn.execute(
                    "DELETE FROM skipped_texts WHERE text_id = ? AND annotator = ?",
                    (selected_text_id, annotator),
                )
                conn.commit()
            set_next_skipped_text_id()
            st.toast("Разметка сохранена. Загружается следующий текст...")
            st.session_state.scroll_to_top = True
            st.rerun()

with col_skip:
    if st.button("Пропустить", disabled=show_skipped):
        if not annotator:
            st.error("Укажите имя разметчика в боковой панели.")
        else:
            with connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO skipped_texts (text_id, annotator, created_at) VALUES (?, ?, ?)",
                    (selected_text_id, annotator, datetime.now().isoformat()),
                )
                conn.commit()
            set_next_skipped_text_id()
            st.toast("Текст пропущен. Загружается следующий...")
            st.session_state.scroll_to_top = True
            st.rerun()
