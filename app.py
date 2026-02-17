"""TextsAnnotation - Streamlit app for text annotation (Refactored).

This is the main entry point for the annotation interface.
Uses the new modular architecture with services and repositories.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import extra_streamlit_components as stx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Import new architecture components
from src.models.annotator import Annotator
from src.models.candidate import Candidate
from src.models.intent import Intent
from src.repositories.intent_repo import IntentRepository
from src.services.annotation_service import AnnotationService
from src.services.auth_service import AuthService
from src.services.import_service import ImportService
from src.utils.config import settings
from src.utils.database import init_database, restore_from_dump, start_auto_dump
from src.utils.logger import logger

# Page configuration
st.set_page_config(
    page_title="TextsAnnotation",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize cookie manager
cookie_manager = stx.CookieManager()


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
        "import": ImportService(settings.db_path),
        "intent_repo": IntentRepository(settings.db_path)
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


def handle_import(services: dict, intents: Dict[str, Intent]):
    """Handle CSV import."""
    import_service: ImportService = services["import"]
    intent_repo: IntentRepository = services["intent_repo"]
    
    # Get current versions
    current_model_version = int(intent_repo.get_setting("current_model_version", "0"))
    current_data_version = int(intent_repo.get_setting("current_data_version", "0"))
    
    count = import_service.import_from_csv(
        csv_path=settings.import_csv_path,
        intents=intents,
        top_k=settings.top_k,
        model_version=current_model_version,
        data_version=current_data_version
    )
    
    if count > 0:
        logger.info(f"Imported {count} texts from CSV")
        st.success(f"✅ Импортировано {count} текстов")
        st.cache_resource.clear()  # Clear cache to reload
    else:
        st.info("ℹ️ Нет новых текстов для импорта")


def authenticate_user(auth_service: AuthService) -> Optional[Annotator]:
    """Handle user authentication in sidebar."""
    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = None
    
    # Check cookie if not authenticated
    if not st.session_state.authenticated_user:
        cookie_user = cookie_manager.get("annotator_user")
        if cookie_user:
            user = auth_service.get_annotator(cookie_user)
            if user:
                st.session_state.authenticated_user = user
                logger.info(f"Restored session for user: {user.name}")
    
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
            submitted = st.form_submit_button("Войти", use_container_width=True)
            
            if submitted:
                annotator = auth_service.authenticate(username, password)
                if annotator:
                    st.session_state.authenticated_user = annotator
                    # Set cookie for 30 days
                    cookie_manager.set("annotator_user", annotator.name, expires_at=datetime.now() + timedelta(days=30))
                    st.success(f"✅ Добро пожаловать, {annotator.name}!")
                    st.rerun()
                else:
                    st.error("❌ Неверный пароль")
    
    # Show welcome message in main area
    st.title("📝 TextsAnnotation")
    st.info("👈 Войдите, используя форму слева")
    st.stop()


def show_annotation_interface(
    annotator: Annotator,
    annotation_service: AnnotationService,
    intents: Dict[str, Intent]
):
    """Main annotation interface."""
    # Initialize scroll state
    if "scroll_to_top" not in st.session_state:
        st.session_state.scroll_to_top = False
        
    # Scroll to top if flag is set
    if st.session_state.scroll_to_top:
        js = """
        <script>
            var body = window.parent.document.querySelector(".main");
            if (body) body.scrollTop = 0;
        </script>
        """
        components.html(js, height=0)
        st.session_state.scroll_to_top = False

    st.title("📝 Разметка текстов")
    
    # Sidebar
    with st.sidebar:
        st.write(f"**Пользователь:** {annotator.name}")
        st.write(f"**Язык:** {annotator.language}")
        
        st.write("**Кластеры:**")
        if annotator.clusters:
            for cluster in annotator.clusters:
                st.markdown(f"- {cluster}")
        else:
            st.write("все")
        
        if st.button("🚪 Выйти"):
            st.session_state.authenticated_user = None
            cookie_manager.delete("annotator_user")
            st.rerun()
        
        st.divider()
        
        # Progress
        progress = annotation_service.get_progress(
            annotator=annotator.name,
            clusters=annotator.clusters if annotator.clusters else None,
            language=annotator.language
        )
        
        st.metric("📊 Прогресс", f"{progress['done']} / {progress['total']}")
        
        if progress['total'] > 0:
            pct = int(100 * progress['done'] / progress['total'])
            st.progress(pct / 100, text=f"{pct}%")
        
        st.divider()
        
        # Show skipped toggle
        show_skipped = st.checkbox("🔄 Показать пропущенные", value=False)
    
    # Get all texts for navigation
    all_texts = annotation_service.get_all_texts(
        annotator=annotator.name,
        clusters=annotator.clusters if annotator.clusters else None,
        language=annotator.language
    )
    
    if not all_texts:
        st.success("🎉 Нет текстов для разметки!")
        st.balloons()
        return

    # Initialize session state for current index
    if "current_text_index" not in st.session_state:
        # Find first unannotated
        for i, t in enumerate(all_texts):
            if not t["is_annotated"]:
                st.session_state.current_text_index = i
                break
        else:
            st.session_state.current_text_index = 0

    # Ensure index is within bounds (e.g. after filter change)
    if st.session_state.current_text_index >= len(all_texts):
        st.session_state.current_text_index = 0
        
    # Sidebar navigation dropdown
    st.sidebar.divider()
    st.sidebar.write("🔎 **Навигация**")
    
    # Filters
    col_f1, col_f2, col_f3 = st.sidebar.columns(3)
    with col_f1:
        show_annotated = st.checkbox("✅", value=True, help="Размеченные")
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
        is_skipped = t["is_skipped"]
        
        # Determine status
        if is_skipped:
            if show_skipped:
                add_to_filtered(t, i)
        elif is_annotated:
            if show_annotated:
                add_to_filtered(t, i)
        else:
            if show_pending:
                add_to_filtered(t, i)
    
    # Ensure current text is always in the list to prevent switching
    current_real_id = all_texts[st.session_state.current_text_index]["id"]
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
        is_skipped = t["is_skipped"]
        is_current = (t["id"] == current_real_id)
        
        should_show = False
        if is_current:
            should_show = True
        elif is_skipped:
            if show_skipped:
                should_show = True
        elif is_annotated:
            if show_annotated:
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

    # Create options list for selectbox
    # Format: "ID: [Status] Text..."
    nav_options = []
    for t in filtered_texts:
        if t["is_skipped"]:
            status = "⏭️"
        elif t["is_annotated"]:
            status = "✅"
        else:
            status = "⬜️"
            
        text_preview = t["request_text"][:30] + "..." if len(t["request_text"]) > 30 else t["request_text"]
        nav_options.append(f"{t['id']}: {status} {text_preview}")
    
    # Determine safe index in FILTERED list
    current_filtered_index = 0
    # current_real_id is already known
    
    for idx, t in enumerate(filtered_texts):
        if t["id"] == current_real_id:
            current_filtered_index = idx
            break
    
    selected_nav = st.sidebar.selectbox(
        "Перейти к тексту:",
        options=nav_options,
        index=current_filtered_index,
        key="nav_selectbox"
    )
    
    # Update global index based on selection
    # Need to handle case where selection triggers rerun
    # But scroll should only happen on Save/Skip
    
    selected_filtered_index = nav_options.index(selected_nav)
    original_index = text_to_original_index[selected_filtered_index]
    
    if original_index != st.session_state.current_text_index:
        st.session_state.current_text_index = original_index
        # st.rerun() # Removed as per instruction

    # Get current text
    current_text = all_texts[st.session_state.current_text_index]
    text_id = current_text["id"]
    text_content = current_text["request_text"]
    is_completed = current_text["is_annotated"]
    is_skipped_status = current_text["is_skipped"]

    # Navigation arrows (Main Area)
    col_prev, col_status, col_next = st.columns([1, 4, 1])
    
    with col_prev:
        if st.button("⬅️", use_container_width=False, disabled=st.session_state.current_text_index == 0):
            st.session_state.current_text_index -= 1
            st.rerun()
            
    with col_next:
        if st.button("➡️", use_container_width=False, disabled=st.session_state.current_text_index == len(all_texts) - 1):
            st.session_state.current_text_index += 1
            st.rerun()
            
    with col_status:
        # Centered text ID
        st.markdown(f"<div style='text-align: center; margin-bottom: 5px;'>Текст {st.session_state.current_text_index + 1} из {len(all_texts)}</div>", unsafe_allow_html=True)
        
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
    st.subheader("Текст для разметки:")
    st.info(text_content)
    
    # Get candidates
    text_obj, candidates = annotation_service.get_text_with_candidates(text_id)
    
    # Filter candidates by cluster
    filtered_candidates = [
        c for c in candidates
        if not annotator.clusters or intents.get(c.label, Intent(label=c.label)).cluster in annotator.clusters
    ]
    
    # Collect decisions
    st.subheader("Выберите подходящие интенты:")
    
    # Inject JS for arrow key navigation
    # This script finds all checkboxes and adds keyboard navigation
    js_script = """
    <script>
    const checkboxes = Array.from(window.parent.document.querySelectorAll('div[data-testid="stCheckbox"] input'));
    checkboxes.forEach((cb, index) => {
        cb.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                const next = checkboxes[index + 1];
                if (next) next.focus();
            }
            if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                const prev = checkboxes[index - 1];
                if (prev) prev.focus();
            }
        });
    });
    </script>
    """
    components.html(js_script, height=0, width=0)
    
    decisions = {}
    candidate_labels = [c.label for c in filtered_candidates]
    shown_intents_source = {}
    
    for candidate in filtered_candidates[:settings.top_k]:
        intent = intents.get(candidate.label)
        if not intent:
            continue
        
        pct = int(candidate.probability * 100)
        
        # Format label
        label_text = f"**{candidate.label}** ({pct}%)"
        
        # Remove tooltip/help as requested
        decision = st.checkbox(
            label_text, 
            key=f"cand_{candidate.label}_{text_id}" # Unique key per text to reset state
        )
        
        if intent.description:
            st.caption(f"└─ {intent.description}")
            
        decisions[candidate.label] = "yes" if decision else "no"
        shown_intents_source[candidate.label] = "candidate"
        st.write("") # Add spacing
    
    # Extra intents
    st.divider()
    available_intents = [label for label, intent in intents.items()
                        if (not annotator.clusters or intent.cluster in annotator.clusters)
                        and label not in candidate_labels]
    
    # Custom formatter with bold intent names
    def format_intent_option(label):
        intent = intents.get(label)
        desc = f" - {intent.description}" if intent and intent.description else ""
        return f"{to_bold(label)}{desc}"

    extra_labels = st.multiselect(
        "Добавить дополнительные интенты:",
        options=available_intents,
        format_func=format_intent_option
    )
    
    for extra in extra_labels:
        shown_intents_source[extra] = "extra"
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    def find_next_pending_index(current_idx, all_texts):
        # Search forward
        for i in range(current_idx + 1, len(all_texts)):
            if not all_texts[i]["is_annotated"] and not all_texts[i]["is_skipped"]:
                return i
        # Wrap around from start
        for i in range(0, current_idx):
            if not all_texts[i]["is_annotated"] and not all_texts[i]["is_skipped"]:
                return i
        # If no strict pending found, try to find ANY unannotated (including skipped)
        for i in range(current_idx + 1, len(all_texts)):
            if not all_texts[i]["is_annotated"]:
                return i
        for i in range(0, current_idx):
            if not all_texts[i]["is_annotated"]:
                return i
                
        # If all done, stay current
        return min(current_idx + 1, len(all_texts) - 1)

    with col1:
        if st.button("✅ Сохранить", type="primary", use_container_width=True):
            annotation_service.save_annotations(
                text_id=text_id,
                annotator=annotator.name,
                decisions=decisions,
                candidate_labels=candidate_labels,
                extra_labels=extra_labels,
                shown_intents_source=shown_intents_source
            )
            st.success("Сохранено!")
            
            # Jump to next pending text and SCROLL TOP
            next_idx = find_next_pending_index(st.session_state.current_text_index, all_texts)
            st.session_state.current_text_index = next_idx
            st.session_state.scroll_to_top = True
            st.rerun()
    
    with col2:
        if st.button("⏭️ Пропустить", use_container_width=True):
            annotation_service.skip_text(text_id, annotator.name)
            st.info("Текст пропущен")
            
            # Jump to next pending text and SCROLL TOP
            next_idx = find_next_pending_index(st.session_state.current_text_index, all_texts)
            st.session_state.current_text_index = next_idx
            st.session_state.scroll_to_top = True
            st.rerun()
    
    with col3:
        if show_skipped and st.button("🔄 Вернуть в работу", use_container_width=True):
            annotation_service.unskip_text(text_id, annotator.name)
            st.info("Текст возвращён в работу")
            st.rerun()


def main():
    """Main application entry point."""
    # Initialize
    intents = initialize_app()
    services = get_services()
    
    # Handle import
    if os.path.exists(settings.import_csv_path):
        if "import_done" not in st.session_state:
            handle_import(services, intents)
            st.session_state.import_done = True
    
    # Authenticate
    auth_service: AuthService = services["auth"]
    annotator = authenticate_user(auth_service)
    
    # Show annotation interface
    annotation_service: AnnotationService = services["annotation"]
    show_annotation_interface(annotator, annotation_service, intents)


if __name__ == "__main__":
    main()
