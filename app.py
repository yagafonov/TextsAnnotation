"""TextsAnnotation - Streamlit app for text annotation (Refactored).

This is the main entry point for the annotation interface.
Uses the new modular architecture with services and repositories.
"""

import sys
import html
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from streamlit_cookies_controller import CookieController
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env", override=True)

_MOD_KEY = "⌘" if sys.platform == "darwin" else "Ctrl"

# Import new architecture components
from src.models.annotator import Annotator
from src.models.candidate import Candidate
from src.models.intent import Intent
from src.repositories.intent_repo import IntentRepository
from src.repositories.user_settings_repo import UserSettingsRepository
from src.services.annotation_service import AnnotationService
from src.services.auth_service import AuthService
from src.utils.config import settings, LANGUAGE_NAMES
from src.utils.database import init_database, restore_from_dump, start_auto_dump
from src.utils.logger import logger

# Read layout preference from query params (before set_page_config)
_layout_pref = st.query_params.get("layout", "wide")
if _layout_pref not in ("wide", "centered"):
    _layout_pref = "wide"

# Page configuration
st.set_page_config(
    page_title="TextsAnnotation",
    page_icon="📝",
    layout=_layout_pref,
    initial_sidebar_state="expanded"
)

# Dark theme CSS (injected early to minimize flash)
DARK_THEME_CSS = """
<style>
    :root {
        --primary-color: #ff4b4b;
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #fafafa;
        color-scheme: dark;
    }

    /* ── Base ─────────────────────────────────────────────── */
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"]  { background-color: #262730; }
    [data-testid="stHeader"]   { background-color: #0e1117; }

    /* ── Text / labels ───────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stMarkdown"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stMarkdown"], label,
    .stCheckbox label, .stCheckbox span,
    [data-testid="stText"], [data-testid="stCaption"],
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"],
    p, span, div { color: #fafafa; }

    /* ── Buttons (secondary / ghost) ─────────────────────── */
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-borderless"],
    button[kind="secondary"], button[kind="borderless"] {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: #4a4a5a !important;
    }
    [data-testid="stBaseButton-secondary"]:hover,
    button[kind="secondary"]:hover {
        background-color: #3a3a4a !important;
        border-color: #6a6a7a !important;
    }

    /* ── Inputs ──────────────────────────────────────────── */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #262730 !important; color: #fafafa !important;
        border-color: #4a4a5a !important;
    }

    /* ── Select / Multiselect controls ───────────────────── */
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        border-color: #4a4a5a !important;
    }
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stMultiSelect div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] div {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    /* Popover dropdown list */
    div[data-baseweb="popover"] { background-color: #262730 !important; }
    div[data-baseweb="popover"] ul  { background-color: #262730 !important; }
    div[data-baseweb="popover"] li  { color: #fafafa !important; }
    div[data-baseweb="popover"] li:hover { background-color: #3a3a4a !important; }
    /* Selected tags in multiselect */
    [data-baseweb="tag"] { background-color: #3a3a4a !important; color: #fafafa !important; }
    /* Search input inside multiselect / select dropdown */
    div[data-baseweb="popover"] input,
    div[data-baseweb="select"] input {
        background-color: #1e2130 !important;
        color: #fafafa !important;
    }
    div[data-baseweb="popover"] input::placeholder,
    div[data-baseweb="select"] input::placeholder { color: #9a9aaa !important; }

    /* ── Progress bar text ───────────────────────────────── */
    [data-testid="stProgressBar"] + div,
    [data-testid="stProgress"] div, .stProgress div,
    div[role="progressbar"] + p { color: #fafafa !important; }

    /* ── Tooltip / help icons ────────────────────────────── */
    [data-testid="stTooltipIcon"] svg path { fill: #9a9aaa !important; }
    [data-testid="stTooltipContent"] { background-color: #262730 !important; color: #fafafa !important; }

    /* ── Misc ────────────────────────────────────────────── */
    hr { border-color: #4a4a5a; }
    [data-testid="stExpander"] { border-color: #4a4a5a; }
    .stDataFrame, .stTable { color: #fafafa; }
    [data-testid="stNotification"] { background-color: #262730 !important; }
    /* Intent blocks (bordered containers in body) */
    div.stMainBlockContainer [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
        background-color: #262730 !important;
        border-color: #3a3a4a !important;
    }

    /* ── Dark-theme overrides for action buttons ─────────── */
    div[data-testid="stHorizontalBlock"]:has(button[kind="primary"]) {
        background-color: #0e1117 !important;
        border-top-color: rgba(250, 250, 250, 0.1) !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: #4a4a5a !important;
    }
</style>
"""

# Always-on CSS: action buttons styling (works in both light and dark themes)
st.markdown("""
<style>
    /* ── Action Buttons: sticky at bottom of viewport ───── */
    div[data-testid="stHorizontalBlock"]:has(button[kind="primary"]) {
        position: sticky;
        bottom: 0px;
        z-index: 999;
        background-color: #ffffff;
        padding: 1rem 0;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
    }
    /* ── Hide injected JS iframes from layout flow ──────── */
    [data-testid="stHtml"] {
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

if st.query_params.get("dark") == "1":
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

cookie_manager = CookieController()


def _sync_settings_from_cookies():
    """Restore theme/layout settings from cookies into query params when absent from URL."""
    needs_rerun = False
    ck_dark = cookie_manager.get("dark_mode")
    ck_layout = cookie_manager.get("layout_mode")
    if ck_dark is not None and "dark" not in st.query_params:
        st.query_params["dark"] = ck_dark
        needs_rerun = True
    if ck_layout is not None and "layout" not in st.query_params:
        st.query_params["layout"] = ck_layout
        needs_rerun = True
    if needs_rerun:
        st.rerun()


@st.cache_resource
def initialize_app():
    """Initialize application resources."""
    # Database setup
    restore_from_dump(settings.db_path, settings.db_dump_path)
    init_database(settings.db_path)
    start_auto_dump(settings.db_path, settings.db_dump_path, settings.db_dump_interval)
    
    # Load intents and sync to database
    intent_repo = IntentRepository(settings.db_path)
    intents = intent_repo.load_from_yaml(settings.intents_path)
    
    for intent in intents.values():
        intent_repo.upsert(intent)
    
    logger.info(f"Application initialized with {len(intents)} intents")
    return intents


@st.cache_resource
def get_services():
    """Get service instances."""
    return {
        "auth": AuthService(settings.annotators_path),
        "annotation": AnnotationService(settings.db_path),
        "intent_repo": IntentRepository(settings.db_path),
        "user_settings": UserSettingsRepository(settings.db_path),
    }


def to_bold(text: str) -> str:
    """Convert text to unicode bold sans-serif."""
    # Mathematical Bold Sans-Serif
    # A-Z: 1D5D4-1D5ED
    # a-z: 1D5EE-1D607
    # 0-9: 1D7EC-1D7F5
    result = ""
    for char in text:
        if "A" <= char <= "Z":
            result += chr(ord(char) - ord("A") + 0x1D5D4)
        elif "a" <= char <= "z":
            result += chr(ord(char) - ord("a") + 0x1D5EE)
        elif "0" <= char <= "9":
            result += chr(ord(char) - ord("0") + 0x1D7EC)
        else:
            result += char
    return result



def authenticate_user(auth_service: AuthService) -> Optional[Annotator]:
    """Handle user authentication in sidebar."""
    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = None
    
    # Check query params if not authenticated
    if not st.session_state.authenticated_user:
        q_user = st.query_params.get("user")
        if q_user:
            user = auth_service.get_annotator(q_user)
            if user:
                st.session_state.authenticated_user = user
                logger.info(f"Restored session from URL for user: {user.name}")
    
    # Check cookie if still not authenticated
    if not st.session_state.authenticated_user:
        cookie_user = cookie_manager.get("annotator_user")
        if cookie_user:
            user = auth_service.get_annotator(cookie_user)
            if user:
                st.session_state.authenticated_user = user
                # Sync back to URL
                st.query_params["user"] = user.name
                logger.info(f"Restored session from cookie for user: {user.name}")
    
    if st.session_state.authenticated_user:
        return st.session_state.authenticated_user

    # Show login in sidebar
    with st.sidebar:
        st.header("🔐 Вход")
        
        # Load annotator names for dropdown
        config = auth_service.load_annotators()
        annotator_names = [a.name for a in config.annotators]
        
        if not annotator_names:
            st.error("❌ Нет доступных пользователей")
            st.stop()
        
        with st.form("login_form"):
            username = st.selectbox(
                "Имя пользователя",
                options=annotator_names,
                help="Выберите ваше имя из списка"
            )
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", width="stretch")
            
            if submitted:
                annotator = auth_service.authenticate(username, password)
                if annotator:
                    st.session_state.authenticated_user = annotator
                    # Set cookie and query param
                    cookie_manager.set("annotator_user", annotator.name, expires=datetime.now() + timedelta(days=30))
                    st.query_params["user"] = annotator.name
                    st.success(f"✅ Добро пожаловать, {annotator.name}!")
                    st.rerun()
                else:
                    st.error("❌ Неверный пароль")
    
    # Show welcome message in main area
    st.title("📝 TextsAnnotation")
    st.info("👈 Войдите, используя форму слева")
    st.stop()


def _apply_user_settings(user_settings_repo: UserSettingsRepository, annotator_name: str) -> None:
    """Load DB settings for annotator and write to query_params where absent."""
    saved = user_settings_repo.load(annotator_name)
    needs_rerun = False
    if saved.get("dark_mode") and "dark" not in st.query_params:
        st.query_params["dark"] = saved["dark_mode"]
        needs_rerun = True
    if saved.get("layout_mode") and "layout" not in st.query_params:
        st.query_params["layout"] = saved["layout_mode"]
        needs_rerun = True
    if saved.get("selected_label") and "label" not in st.query_params:
        st.query_params["label"] = saved["selected_label"]
        needs_rerun = True
    if needs_rerun:
        st.rerun()

def show_annotation_interface(
    annotator: Annotator,
    annotation_service: AnnotationService,
    intents: Dict[str, Intent],
    user_settings_repo: UserSettingsRepository,
):
    """Main annotation interface."""
    # Initialize scroll state
    if "scroll_to_top" not in st.session_state:
        st.session_state.scroll_to_top = False


    
    # Sidebar
    with st.sidebar:
        st.write(f"**Пользователь:** {annotator.name}")
        
        lang_name = LANGUAGE_NAMES.get(annotator.language, annotator.language)
        st.write(f"**Язык:** {lang_name}")
        
        st.write(f"**{'Интенты' if annotator.intents else 'Кластеры'}:**")
        # Build label stats for the dropdown — show all labels with assigned texts
        _use_intents = bool(annotator.intents)
        _assigned_labels = annotation_service.get_assigned_labels(
            annotator=annotator.name,
            by_cluster=not _use_intents,
            threshold=0.0
        )
        _labels = _assigned_labels
        if _labels:
            _stats = annotation_service.get_label_stats(
                annotator=annotator.name,
                labels=tuple(_labels),
                threshold=0.0,
                by_cluster=not _use_intents
            )
            # Add "uncategorized" bucket for texts with no stored candidates
            _uncat = annotation_service.get_uncategorized_count(
                annotator=annotator.name,
                threshold=0.0
            )
            if _uncat["total"] > 0:
                _stats.append({"label": "__other__", "total": _uncat["total"], "annotated": _uncat["annotated"]})

            _stats.sort(key=lambda s: (-(s["total"] - s["annotated"]), s["label"]))
            _options = [None] + [s["label"] for s in _stats]
            _format = {
                None: "Все",
                **{
                    s["label"]: (
                        f"Другое ({s['annotated']}/{s['total']})" if s["label"] == "__other__"
                        else f"{s['label']} ({s['annotated']}/{s['total']})"
                    )
                    for s in _stats
                }
            }
            # FIX 2: restore selected_label from ?label= query param on first load
            if "selected_label" not in st.session_state:
                _q_label = st.query_params.get("label")
                if _q_label in _options:
                    st.session_state["selected_label"] = _q_label
            _prev_label = st.session_state.get("selected_label")
            _selected = st.selectbox(
                "Фильтр",
                options=_options,
                format_func=lambda x: _format.get(x, x),
                key="selected_label",
                label_visibility="collapsed"
            )
            # Sync selected label back to query param
            _cur_label = st.session_state.get("selected_label")
            if _cur_label:
                st.query_params["label"] = _cur_label
            elif "label" in st.query_params:
                del st.query_params["label"]
            if _selected != _prev_label:
                # If filter changed, check if current text still fits; if not, reset
                st.session_state["_label_filter_changed"] = True
                user_settings_repo.save(annotator.name, selected_label=_selected or "")
        else:
            st.write("все")
        
        # Settings row
        _c_logout, _c_dark, _c_layout = st.columns([2, 1, 1])
        with _c_logout:
            if st.button("Выйти", width="stretch"):
                st.session_state.authenticated_user = None
                if cookie_manager.get("annotator_user"):
                    cookie_manager.remove("annotator_user")
                # Clear user param but preserve settings
                if "user" in st.query_params:
                    del st.query_params["user"]
                st.rerun()
        with _c_dark:
            _is_dark = st.query_params.get("dark") == "1"
            if st.button("🌙" if not _is_dark else "☀️", width="stretch", help="Тема"):
                new_val = "0" if _is_dark else "1"
                st.query_params["dark"] = new_val
                cookie_manager.set("dark_mode", new_val, expires=datetime.now() + timedelta(days=365))
                user_settings_repo.save(annotator.name, dark_mode=new_val)
                st.rerun()
        with _c_layout:
            _is_wide = st.query_params.get("layout", "wide") == "wide"
            if st.button("↔️" if not _is_wide else "↕️", width="stretch", help="Широкий/узкий режим"):
                new_val = "centered" if _is_wide else "wide"
                st.query_params["layout"] = new_val
                cookie_manager.set("layout_mode", new_val, expires=datetime.now() + timedelta(days=365))
                user_settings_repo.save(annotator.name, layout_mode=new_val)
                st.rerun()

        st.divider()
        # (Progress shown below, after all_texts is computed)

    # Get all texts for navigation (respecting selected label filter)
    # FIX 3: compute navigation list FIRST so progress uses the same set of texts.
    _selected_label = st.session_state.get("selected_label")
    _is_other_filter = _selected_label == "__other__"
    _use_intents = bool(annotator.intents)
    _filter_intents = annotator.intents if annotator.intents else None
    _filter_clusters = annotator.clusters if not annotator.intents and annotator.clusters else None

    all_texts = annotation_service.get_all_texts(
        annotator=annotator.name,
        clusters=_filter_clusters,
        intents=_filter_intents,
        language=annotator.language,
        candidate_label=_selected_label if not _is_other_filter else None,
        candidate_threshold=0.0 if _selected_label and not _is_other_filter else 0.0,
        candidate_by_cluster=not _use_intents,
        uncategorized_threshold=0.0 if _is_other_filter else None
    )


    # Progress: all texts assigned to this annotator (matches navigation list)
    _global_progress = annotation_service.get_progress(
        annotator=annotator.name,
        language=annotator.language
    )
    with st.sidebar:
        _g_done = _global_progress["done"]
        _g_total = _global_progress["total"]
        st.metric("📊 Прогресс", f"{_g_done} / {_g_total}")
        if _g_total > 0:
            _pct = int(100 * _g_done / _g_total)
            st.progress(_pct / 100, text=f"{_pct}%")
        _scope_word = "интенты" if annotator.intents else "кластеры"
        st.caption(f"размечено / назначено (все {_scope_word})")
    
    if not all_texts:
        st.success("🎉 Нет текстов для разметки!")
        st.balloons()
        return

    all_text_ids = [t["id"] for t in all_texts]

    # Initialize session state for current index
    if "current_text_index" not in st.session_state:
        for i, t in enumerate(all_texts):
            if not t["is_annotated"]:
                st.session_state.current_text_index = i
                break
        else:
            st.session_state.current_text_index = 0

    # Handle filter change: stay on current text if it still matches, else go to first
    if st.session_state.pop("_label_filter_changed", False):
        current_id = all_texts[min(st.session_state.current_text_index, len(all_texts) - 1)]["id"] \
            if all_texts else None
        # This current_id is from the OLD all_texts (before filter). Check if it's in the new list.
        # We need to check against original (pre-filter) list - but we can check session text id.
        _cur_id_in_session = st.session_state.get("current_text_id")
        if _cur_id_in_session and _cur_id_in_session in all_text_ids:
            st.session_state.current_text_index = all_text_ids.index(_cur_id_in_session)
        else:
            # Jump to first unannotated in the filtered list
            for i, t in enumerate(all_texts):
                if not t["is_annotated"]:
                    st.session_state.current_text_index = i
                    break
            else:
                st.session_state.current_text_index = 0

    # Ensure index is within bounds (e.g. after filter change)
    if st.session_state.current_text_index >= len(all_texts):
        st.session_state.current_text_index = 0

    # Universal Scroll to Top on Text Change (Now safe after initialization)
    if all_texts:
        _curr_id = all_texts[min(st.session_state.current_text_index, len(all_texts)-1)]["id"]
        if st.session_state.get("last_viewed_text_id") != _curr_id:
            st.session_state.scroll_to_top = True
            st.session_state.last_viewed_text_id = _curr_id

    if st.session_state.scroll_to_top:
        js = """
        <script>
        (function() {
            function scrollUp() {
                var selectors = [
                    "section.stMain",
                    "[data-testid='stAppViewContainer']",
                    ".main",
                    "section.main"
                ];
                for (var i = 0; i < selectors.length; i++) {
                    var el = window.parent.document.querySelector(selectors[i]);
                    if (el) { el.scrollTop = 0; }
                }
                window.parent.scrollTo(0, 0);
            }
            scrollUp();
            setTimeout(scrollUp, 100);
            setTimeout(scrollUp, 300);
        })();
        </script>
        """
        components.html(js, height=0)
        st.session_state.scroll_to_top = False

        
    # Sidebar navigation dropdown
    st.sidebar.divider()
    st.sidebar.write("🔎 **Навигация**")
    
    # Filters
    col_f1, col_f2, col_f3 = st.sidebar.columns(3)
    with col_f1:
        show_annotated = st.checkbox("✅", value=False, help="Размеченные")
    with col_f2:
        show_pending = st.checkbox("⬜️", value=True, help="Ожидающие")
    with col_f3:
        show_skipped = st.checkbox("⏭️", value=True, help="Пропущенные")
    
    # Filter texts
    filtered_texts = []
    text_to_original_index = {} # Map filtered index to original index in all_texts
    
    # helper to add text
    def add_to_filtered(t, original_idx):
        filtered_texts.append(t)
        text_to_original_index[len(filtered_texts)-1] = original_idx

    for i, t in enumerate(all_texts):
        is_annotated = t["is_annotated"]
        is_skipped = t["is_skipped"] and not is_annotated

        # Determine status: annotated takes priority over skipped
        if is_annotated:
            if show_annotated:
                add_to_filtered(t, i)
        elif is_skipped:
            if show_skipped:
                add_to_filtered(t, i)
        else:
            if show_pending:
                add_to_filtered(t, i)
    
    # Ensure current text is always in the list to prevent switching
    current_real_id = all_texts[st.session_state.current_text_index]["id"]
    st.session_state["current_text_id"] = current_real_id  # Used for filter-change navigation
    current_in_list = False
    for t in filtered_texts:
        if t["id"] == current_real_id:
            current_in_list = True
            break
            
    if not current_in_list:
        # Add current text to list (filtered_texts)
        # We need to find its original index
        # We know st.session_state.current_text_index matches all_texts
        curr_t = all_texts[st.session_state.current_text_index]
        # Insert in correct order or just append?
        # Appending is safe. Or insert based on ID?
        # Let's insert based on order in all_texts to maintain sort
        # But that requires re-building.
        # Simple fix: Append if not present.
        add_to_filtered(curr_t, st.session_state.current_text_index)
        # Re-sort filtered list by ID to match general order?
        # The main loop pushed in order. The only out-of-order item could be this one.
        # But users might expect ID order.
        # Let's re-sort filtered_texts and rebuild map?
        # This is getting complex.
        # Simpler: Modify the MAIN loop to ALWAYS include current_real_id.
        pass

    # Re-run loop with forced inclusion?
    # Better: Reset and do it right.
    filtered_texts = []
    text_to_original_index = {} 
    
    for i, t in enumerate(all_texts):
        is_annotated = t["is_annotated"]
        is_skipped = t["is_skipped"] and not is_annotated
        is_current = (t["id"] == current_real_id)

        should_show = False
        if is_current:
            should_show = True
        elif is_annotated:
            if show_annotated:
                should_show = True
        elif is_skipped:
            if show_skipped:
                should_show = True
        else:
            if show_pending:
                should_show = True
                
        if should_show:
            filtered_texts.append(t)
            text_to_original_index[len(filtered_texts)-1] = i

    if not filtered_texts:
        st.sidebar.info("Нет текстов, соответствующих фильтрам")
        st.info("Выберите хотя бы один фильтр слева или измените параметры")
        return

    # Determine safe index in FILTERED list
    current_filtered_index = 0
    # current_real_id is already known
    

    for idx, t in enumerate(filtered_texts):
        if t["id"] == current_real_id:
            current_filtered_index = idx
            break
    
    # NAVIGATION: Searchable dropdown
    # Create options for selectbox: list of (index_in_filtered, text_representation)
    # We use index because text IDs might not be contiguous or sorted heavily
    # But for display we want "ID: Text..."
    
    # We want the selectbox to return the index in filtered_texts
    # format_func will handle display
    
    options = list(range(len(filtered_texts)))
    
    def format_text_option(idx):
        t = filtered_texts[idx]
        is_current = (idx == current_filtered_index)
        marker = "📍 " if is_current else ""
        status = ""
        if t["is_annotated"]: status = "✅ "
        elif t["is_skipped"]: status = "⏭️ "
        
        # Truncate text
        text_preview = t["text"][:40] + "..." if len(t["text"]) > 40 else t["text"]
        return f"{marker}{status}[{t['id']}] {text_preview}"

    def on_text_nav_change():
        selected = st.session_state["nav_text_select"]
        if selected in text_to_original_index:
            st.session_state.current_text_index = text_to_original_index[selected]

    # Sync selectbox to current text before rendering (must be set before widget creation)
    st.session_state["nav_text_select"] = current_filtered_index

    st.sidebar.selectbox(
        "Перейти к тексту:",
        options=options,
        format_func=format_text_option,
        key="nav_text_select",
        on_change=on_text_nav_change,
        help="Выберите текст из списка или начните ввод для поиска"
    )
    st.sidebar.info(f"Текст ID: {current_real_id} ({current_filtered_index + 1} из {len(filtered_texts)} отфильтрованных)")

    # Get current text
    current_text = all_texts[st.session_state.current_text_index]
    text_id = current_text["id"]
    text_content = current_text["text"]
    is_completed = current_text["is_annotated"]
    is_skipped_status = current_text["is_skipped"] and not is_completed

    # Navigation arrows (Main Area) — iterate within filtered list
    col_prev, col_status, col_next = st.columns([1, 4, 1])

    with col_prev:
        if st.button(f"⬅️ + {_MOD_KEY}", width="content", disabled=current_filtered_index == 0):
            prev_filtered = current_filtered_index - 1
            st.session_state.current_text_index = text_to_original_index[prev_filtered]
            st.rerun()

    with col_next:
        if st.button(f"{_MOD_KEY} + ➡️", width="content", disabled=current_filtered_index >= len(filtered_texts) - 1):
            next_filtered = current_filtered_index + 1
            st.session_state.current_text_index = text_to_original_index[next_filtered]
            st.rerun()

    with col_status:
        # Centered text counter + filter description
        _scope_label = st.session_state.get("selected_label")
        _scope_kind  = "Интент" if annotator.intents else "Кластер"
        _label_part  = f"{_scope_kind}: {_scope_label}. " if _scope_label else ""

        # Build status description from checkbox flags
        _flags = (show_annotated, show_pending, show_skipped)
        _status_desc = {
            (True,  True,  True ): "все",
            (False, True,  False): "только неразмеченные без пропусков",
            (False, True,  True ): "неразмеченные",
            (False, False, True ): "только пропущенные",
            (True,  False, False): "только размеченные",
            (True,  True,  False): "все без пропусков",
            (True,  False, True ): "размеченные и пропущенные",
            (False, False, False): "нет фильтров",
        }.get(_flags, "все")

        st.markdown(
            f"<div style='text-align:center;margin-bottom:2px;'>"
            f"Текст {current_filtered_index + 1} из {len(filtered_texts)}</div>"
            f"<div style='text-align:center;font-size:0.78em;opacity:0.6;margin-bottom:5px;'>"
            f"{_label_part}{_status_desc}</div>",
            unsafe_allow_html=True
        )
        
        # Centered status badge
        if is_skipped_status:
            st.markdown(
                """
                <div style='display: flex; justify-content: center;'>
                    <div style='background-color: #e6f3ff; color: #0068c9; padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 14px; border: 1px solid #b3d9ff;'>
                        ⏭️ Этот текст был пропущен
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif is_completed:
            st.markdown(
                """
                <div style='display: flex; justify-content: center;'>
                    <div style='background-color: #d1fae5; color: #065f46; padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 14px; border: 1px solid #a7f3d0;'>
                        ✅ Этот текст уже размечен
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

    # Display text
    # st.info(text_content) replaced with custom markdown to fit content width
    st.markdown(
        f"""
        <div style="background-color: #e6f3ff; color: #004085; padding: 1rem; border-radius: 0.5rem; border: 1px solid #b8daff; width: fit-content; max_width: 100%; margin-bottom: 1rem; font-weight: bold; font-size: 1.1rem;">
            {html.escape(text_content)}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get candidates
    text_obj, candidates = annotation_service.get_text_with_candidates(text_id)

    # Filter candidates by cluster (used for cluster-scoped annotators)
    filtered_candidates = [
        c for c in candidates
        if not annotator.clusters or intents.get(c.label, Intent(label=c.label)).cluster in annotator.clusters
    ]

    # Build shown candidates: top-k by model confidence UNION annotator's intents above threshold
    threshold = settings.annotators_intents_confidence_threshold
    shown_map = {}  # label -> (candidate, source)
    for c in filtered_candidates[:settings.top_k]:
        shown_map[c.label] = (c, "candidate")
    if annotator.intents:
        for c in candidates:
            if c.label in annotator.intents and c.probability >= threshold and c.label not in shown_map:
                shown_map[c.label] = (c, "annotator_intent")
    _show_conf = annotation_service.text_repo.get_setting("show_confidence", "0") == "1"
    if _show_conf:
        shown_items = sorted(shown_map.values(), key=lambda x: x[0].probability, reverse=True)
    else:
        shown_items = sorted(shown_map.values(), key=lambda x: x[0].label)

    # Collect decisions


    # Inject JS for arrow key navigation, auto-focus, and Enter to Save
    js_script = f"""
    <script>
    (function() {{
        const pWin = window.parent;
        const pDoc = pWin.document;
        const currentTextId = "{text_id}";
        
        // --- UTILS ---
        const getCBs = () => {{
            return Array.from(pDoc.querySelectorAll('div[data-testid="stCheckbox"] input'))
                        .filter(cb => !cb.closest('[data-testid="stSidebar"]'));
        }};

        const focusCB = (index) => {{
            const cbs = getCBs();
            if (cbs[index]) {{
                cbs[index].focus();
                pWin._lastFocusedIndex = index;
            }}
        }};

        // --- RESET ON NEW TEXT ---
        if (pWin._lastTextId !== currentTextId) {{
            pWin._lastFocusedIndex = 0;
            pWin._lastTextId = currentTextId;
        }}

        // --- HANDLER ---
        const shortcutHandler = (e) => {{
            const active = pDoc.activeElement;
            const isInput = active.tagName === 'TEXTAREA' || (active.tagName === 'INPUT' && active.type !== 'checkbox');
            
            // 1. Space bar (Global scroll prevention)
            if ((e.key === ' ' || e.code === 'Space') && !isInput) {{
                e.preventDefault();
                e.stopImmediatePropagation();
                if (active.type === 'checkbox') {{
                    active.click();
                }}
                return false;
            }}

            // 2. Enter key
            if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) {{
                const isMultiselect = active.tagName === 'INPUT' && active.closest('.stMultiSelect');
                const isEmptyMultiselect = isMultiselect && active.value === '';

                if ((!isInput || isEmptyMultiselect) && active.tagName !== 'BUTTON') {{
                    const btn = pDoc.querySelector('button[kind="primary"]');
                    if (btn) {{
                        e.preventDefault();
                        e.stopImmediatePropagation();
                        btn.click();
                    }}
                }}
            }}

            // 2b. Escape key — Skip / Unskip
            if (e.key === 'Escape' && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) {{
                if (!isInput) {{
                    const btns = Array.from(pDoc.querySelectorAll('button'));
                    const skipBtn = btns.find(b => b.textContent.includes('Пропустить'));
                    if (skipBtn) {{
                        e.preventDefault();
                        e.stopImmediatePropagation();
                        skipBtn.click();
                    }}
                }}
            }}

            // 3. Ctrl+Arrow / Cmd+Arrow — prev/next text
            if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && (e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {{
                const btns = Array.from(pDoc.querySelectorAll('button'));
                const target = e.key === 'ArrowLeft'
                    ? btns.find(b => b.textContent.includes('⬅'))
                    : btns.find(b => b.textContent.includes('➡'));
                if (target && !target.disabled) {{
                    e.preventDefault();
                    e.stopImmediatePropagation();
                    target.click();
                }}
                return false;
            }}

            // 3.1 Right Arrow (toggle examples)
            if (e.key === 'ArrowRight' && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey) {{
                if (active.type === 'checkbox') {{
                    // stVerticalBlock is the container for the group (checkbox + caption + expander)
                    const container = active.closest('div[data-testid="stVerticalBlock"]');
                    if (container) {{
                        const expander = container.querySelector('[data-testid="stExpander"] summary');
                        if (expander) {{
                            e.preventDefault();
                            e.stopImmediatePropagation();
                            expander.click();
                        }}
                    }}
                }}
            }}

            // 4. Arrow Keys (checkbox navigation)
            if (['ArrowUp', 'ArrowDown'].includes(e.key)) {{
                if (active.type === 'checkbox') {{
                    const cbs = getCBs();
                    const idx = cbs.indexOf(active);
                    if (idx !== -1) {{
                        let nextIdx = idx;
                        if (e.key === 'ArrowDown') nextIdx++;
                        if (e.key === 'ArrowUp') nextIdx--;
                        
                        if (nextIdx >= 0 && nextIdx < cbs.length) {{
                            e.preventDefault();
                            e.stopImmediatePropagation();
                            focusCB(nextIdx);
                        }}
                    }}
                }}
            }}
        }};

        // --- ATTACH ---
        if (pWin._shortcutHandler) {{
            pWin.removeEventListener('keydown', pWin._shortcutHandler, true);
        }}
        pWin._shortcutHandler = shortcutHandler;
        pWin.addEventListener('keydown', shortcutHandler, true);

        // --- INITIAL FOCUS ---
        const startFocus = () => {{
            const active = pDoc.activeElement;
            const alreadyFocused = active && active.closest('div[data-testid="stCheckbox"]');
            if (!alreadyFocused) {{
                const targetIdx = pWin._lastFocusedIndex || 0;
                focusCB(targetIdx);
            }}
        }};

        [100, 300, 600, 1000].forEach(delay => setTimeout(startFocus, delay));

        // --- MOBILE: strip shortcut hints from buttons ---
        const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(pWin.navigator.userAgent)
                      || pWin.innerWidth < 768;
        if (isMobile) {{
            const stripHints = () => {{
                pDoc.querySelectorAll('button').forEach(btn => {{
                    btn.textContent = btn.textContent
                        .replace(/\\s*\\[Enter\\]/g, '')
                        .replace(/\\s*\\[Esc\\]/g, '')
                        .replace(/\\s*\\+\\s*⌘/g, '')
                        .replace(/\\s*⌘\\s*\\+\\s*/g, '')
                        .replace(/\\s*\\+\\s*Ctrl/g, '')
                        .replace(/\\s*Ctrl\\s*\\+\\s*/g, '');
                }});
            }};
            [100, 300, 600, 1000].forEach(delay => setTimeout(stripHints, delay));
        }}
    }})();
    </script>
    """
    components.html(js_script, height=0, width=0)
    
    decisions = {}
    candidate_labels = list(shown_map.keys())
    shown_intents_source = {}

    _is_dark = st.query_params.get("dark") == "1"
    # Background for intent blocks — targets both wrapper and inner content div
    _block_bg = "#262730" if _is_dark else "#f0f2f6"
    _block_border = "#3a3a4a" if _is_dark else "#d0d4dd"
    st.markdown(
        f"<style>"
        f"div.stMainBlockContainer [data-testid='stLayoutWrapper'] > [data-testid='stVerticalBlock']"
        f"{{background-color:{_block_bg}!important;}}"
        f"</style>",
        unsafe_allow_html=True,
    )

    for candidate, source in shown_items:
        intent = intents.get(candidate.label)
        if not intent:
            continue

        pct = int(candidate.probability * 100)
        label_text = f"**{candidate.label}** ({pct}%)" if _show_conf else f"**{candidate.label}**"

        with st.container(border=True):
            decision = st.checkbox(
                label_text,
                key=f"cand_{candidate.label}_{text_id}"
            )
            if intent.description:
                st.caption(f"**Описание:** {intent.description}")
            
            all_examples = intent.train + intent.test
            if all_examples:
                with st.expander("📝 Примеры"):
                    examples_md = "\n".join([f"- {ex}" for ex in all_examples])
                    st.markdown(examples_md)
        decisions[candidate.label] = "yes" if decision else "no"
        shown_intents_source[candidate.label] = source

    # Extra intents
    available_intents = sorted(
        [label for label in intents if label not in candidate_labels]
    )

    extra_labels = st.multiselect(
        "Добавить дополнительные интенты:",
        options=available_intents
    )
    
    for extra in extra_labels:
        shown_intents_source[extra] = "extra"
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    def find_next_pending_filtered(current_filtered_idx, filtered_texts, text_to_original_index):
        """Find next pending text within the filtered list, return original index."""
        n = len(filtered_texts)
        # Search forward in filtered list
        for i in range(current_filtered_idx + 1, n):
            t = filtered_texts[i]
            if not t["is_annotated"] and not t["is_skipped"]:
                return text_to_original_index[i]
        # Wrap around from start
        for i in range(0, current_filtered_idx):
            t = filtered_texts[i]
            if not t["is_annotated"] and not t["is_skipped"]:
                return text_to_original_index[i]
        # If no strict pending, try ANY unannotated
        for i in range(current_filtered_idx + 1, n):
            if not filtered_texts[i]["is_annotated"]:
                return text_to_original_index[i]
        for i in range(0, current_filtered_idx):
            if not filtered_texts[i]["is_annotated"]:
                return text_to_original_index[i]
        # All done — move to next in filtered list or stay
        next_f = min(current_filtered_idx + 1, n - 1)
        return text_to_original_index[next_f]

    with col1:
        if st.button("✅ Сохранить [Enter]", type="primary", width="stretch"):
            annotation_service.save_annotations(
                text_id=text_id,
                annotator=annotator.name,
                decisions=decisions,
                candidate_labels=candidate_labels,
                extra_labels=extra_labels,
                shown_intents_source=shown_intents_source
            )
            st.success("Сохранено!")
            
            # Jump to next pending text within filtered list and SCROLL TOP
            next_idx = find_next_pending_filtered(current_filtered_index, filtered_texts, text_to_original_index)
            st.session_state.current_text_index = next_idx
            st.rerun()
    
    with col2:
        if st.button("⏭️ Пропустить [Esc]", width="stretch"):
            annotation_service.skip_text(text_id, annotator.name)
            # Jump to next pending text within filtered list
            next_idx = find_next_pending_filtered(current_filtered_index, filtered_texts, text_to_original_index)
            st.session_state.current_text_index = next_idx
            st.rerun()


@st.cache_resource
def ensure_assignments(annotators_hash: str, _annotation_service: AnnotationService, _auth_service: AuthService):
    """Ensure texts are correctly assigned based on current annotators."""
    annotators = _auth_service.load_annotators().annotators
    count = _annotation_service.assign_unannotated_texts(annotators)
    if count > 0:
        logger.info(f"Re-assignment triggered by config change. Updated {count} texts.")

def main():
    """Main application entry point."""
    # Show initialization status only on first load
    _first_load = "app_initialized" not in st.session_state
    if _first_load:
        with st.status("🚀 Запуск приложения...", expanded=True) as status:
            st.write("📂 Подготовка базы данных и интентов...")
            intents = initialize_app()

            st.write("⚙️ Настройка сервисов...")
            services = get_services()

            st.write("⚖️ Балансировка назначений...")
            auth_service: AuthService = services["auth"]
            config = auth_service.load_annotators()
            annotators_repr = "".join(sorted([f"{a.name}:{','.join(sorted(a.intents))}" for a in config.annotators]))
            ensure_assignments(annotators_repr, services["annotation"], auth_service)

            status.update(label="✅ Инициализация завершена", state="complete", expanded=False)
        st.session_state.app_initialized = True
    else:
        intents = initialize_app()
        services = get_services()
        auth_service: AuthService = services["auth"]
        config = auth_service.load_annotators()
        annotators_repr = "".join(sorted([f"{a.name}:{','.join(sorted(a.intents))}" for a in config.annotators]))
        ensure_assignments(annotators_repr, services["annotation"], auth_service)

    _sync_settings_from_cookies()

    annotator = authenticate_user(auth_service)

    # Apply per-user DB settings (cross-device persistence)
    user_settings_repo: UserSettingsRepository = services["user_settings"]
    _apply_user_settings(user_settings_repo, annotator.name)

    # Show annotation interface
    annotation_service: AnnotationService = services["annotation"]
    show_annotation_interface(annotator, annotation_service, intents, user_settings_repo)


if __name__ == "__main__":
    main()
