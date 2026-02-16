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
    
    # Get next text
    text_row = annotation_service.get_next_text(
        annotator=annotator.name,
        clusters=annotator.clusters if annotator.clusters else None,
        language=annotator.language,
        min_annotators=settings.min_annotators,
        show_skipped=show_skipped
    )
    
    if not text_row:
        st.success("✅ Все тексты размечены!")
        st.balloons()
        return
    
    text_id = text_row["id"]
    text_content = text_row["text"]
    
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
        help_text = intent.description if intent.description else None
        
        decision = st.checkbox(
            label_text, 
            key=f"cand_{candidate.label}",
            help=help_text
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
    
    extra_labels = st.multiselect(
        "Добавить дополнительные интенты:",
        options=available_intents,
        format_func=lambda x: f"{x} - {intents[x].description}" if intents[x].description else x
    )
    
    for extra in extra_labels:
        shown_intents_source[extra] = "extra"
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
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
            st.rerun()
    
    with col2:
        if st.button("⏭️ Пропустить", use_container_width=True):
            annotation_service.skip_text(text_id, annotator.name)
            st.info("Текст пропущен")
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
