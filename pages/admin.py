"""Admin Dashboard - Refactored to use StatsService."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import extra_streamlit_components as stx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


from src.repositories.intent_repo import IntentRepository
from src.services.import_service import ImportService
from src.services.stats_service import StatsService
from src.utils.config import settings
from src.utils.database import get_connection

# Admin password
ADMIN_PASSWORD = os.environ.get("TEXTS_ADMIN_PASSWORD", "admin123")

st.set_page_config(page_title="Admin Dashboard", layout="wide", page_icon="📊")

# Initialize cookie manager (must be unique key if multiple on same page, but here it's separate page)
# However, stx.CookieManager key defaults to "init".
cookie_manager = stx.CookieManager(key="admin_cookies")


def authenticate_admin():
    """Admin authentication with session management."""
    # Initialize session state
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
        
    # Check cookie if not authenticated
    if not st.session_state.admin_authenticated:
        # Simple check: if cookie exists and matches password
        # Ideally we should store a hash, but for now we store the password hash or just a specific token
        # For simplicity in this demo, we'll store a simple token "admin_session_valid"
        # SECURITY WARNING: This is a weak implementation for demo purposes.
        # In production, use proper session management.
        admin_token = cookie_manager.get("admin_token")
        if admin_token == "valid_admin_session":
            st.session_state.admin_authenticated = True
    
    # Show logout button if authenticated
    if st.session_state.admin_authenticated:
        with st.sidebar:
            st.success("✅ Администратор авторизован")
            if st.button("🚪 Выйти", width="stretch"):
                st.session_state.admin_authenticated = False
                cookie_manager.delete("admin_token")
                st.rerun()
        return
    
    # Show login form
    with st.sidebar:
        st.header("🔐 Авторизация")
        
        # Callback for authentication
        def try_authenticate():
            if st.session_state.admin_password_input == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.session_state.admin_auth_error = None
                # Set cookie
                cookie_manager.set("admin_token", "valid_admin_session", expires_at=datetime.now() + timedelta(days=30))
            else:
                st.session_state.admin_auth_error = "❌ Неверный пароль"

        st.text_input(
            "Пароль администратора", 
            type="password",
            key="admin_password_input",
            on_change=try_authenticate
        )
        
        if st.session_state.get("admin_auth_error"):
            st.error(st.session_state.admin_auth_error)
            st.session_state.admin_auth_error = None

        if st.button("Войти", width="stretch"):
            try_authenticate()
            if st.session_state.admin_authenticated:
                st.rerun()
    
    # Show instruction in main area
    st.info("👈 Войдите, используя пароль администратора")
    st.stop()


def show_overall_stats(stats_service: StatsService):
    """Display overall statistics."""
    st.header("📊 Общая статистика")
    
    df = stats_service.get_overall_stats()
    
    if not df.empty:
        row = df.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Всего текстов", f"{int(row['total_texts']):,}")
        
        with col2:
            st.metric("Аннотаторов", f"{int(row['total_annotators']):,}")
        
        with col3:
            st.metric("Аннотаций", f"{int(row['total_annotations']):,}")
        
        with col4:
            if row['total_annotations'] > 0:
                yes_pct = int(100 * row['positive_annotations'] / row['total_annotations'])
                st.metric("% Положительных", f"{yes_pct}%")


def show_annotator_stats(stats_service: StatsService):
    """Display annotator statistics."""
    st.header("👥 Статистика аннотаторов")
    
    df = stats_service.get_annotator_stats()
    
    if not df.empty:
        # Format percentages
        df['yes_rate'] = df['yes_rate'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(
            df,
            column_config={
                "annotator": "Аннотатор",
                "texts_annotated": "Размечено текстов",
                "total_decisions": "Всего решений",
                "yes_count": "Да",
                "no_count": "Нет",
                "yes_rate": "% Да"
            },
            hide_index=True,
            width="stretch"
        )
    else:
        st.info("Нет данных")


def show_intent_quality(stats_service: StatsService):
    """Display intent quality metrics."""
    st.header("🎯 Качество интентов")
    
    df = stats_service.get_intent_quality()
    
    if not df.empty:
        # Filter to show only intents with data
        df = df[(df['top1_shown'] > 0) | (df['missed'] > 0)]
        
        # Format percentages
        df['top1_precision'] = df['top1_precision'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )
        df['miss_rate'] = df['miss_rate'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )
        df['annotation_rate'] = df['annotation_rate'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )
        
        # Format confusion percentage for sorting
        df['confusion_percentage_val'] = df['confusion_percentage'].apply(
            lambda x: round(x * 100, 1) if pd.notna(x) else 0.0
        )
        
        # Construct deep dive URL for each row
        # Format: /admin?tab=...&human=...&top5=...
        target_tab = "📝 Тексты"
        df['confusion_link'] = df.apply(
            lambda r: f"/admin?tab={target_tab}&human={r['label']}&top5={r['confusion_label']}&pg=1" 
            if pd.notna(r['confusion_label']) else None, 
            axis=1
        )
        
        st.dataframe(
            df,
            column_config={
                "label": "Интент",
                "cluster": "Кластер",
                "complexity": "Сложность",
                "top1_shown": "Показано как Top-1",
                "top1_accepted": "Принято как Top-1",
                "missed": "Пропущено моделью",
                "top1_precision": "Точность Top-1",
                "miss_rate": "% Пропусков",
                "annotation_rate": "% Разметки",
                "confusion_link": st.column_config.LinkColumn(
                    "Чаще всего спутан с",
                    help="Нажмите на название, чтобы увидеть примеры путаницы в разделе Тексты",
                    display_text=r"top5=([^&]+)" # Extract intent name from URL for display
                ),
                "confusion_percentage_val": st.column_config.NumberColumn(
                    "% Путаницы", 
                    format="%.1f%%",
                    help="Процент случаев, когда модель предсказала этот интент вместо верного"
                )
            },
            column_order=[
                "label", "cluster", "complexity", 
                "top1_shown", "top1_accepted", "top1_precision", 
                "missed", "miss_rate", "annotation_rate", 
                "confusion_link", "confusion_percentage_val"
            ],
            hide_index=True,
            width="stretch"
        )
        
    else:
        st.info("Нет данных о качестве интентов")


def show_text_overview(stats_service: StatsService):
    """Display detailed text overview with filters and pagination."""
    st.header("📝 Обзор текстов")
    
    # All metadata for filters
    all_intents = stats_service.get_all_intents()
    all_annotators = stats_service.get_unique_assigned_annotators()
    
    # 1. Filters Section
    with st.expander("🔍 Фильтры", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            search_query = st.text_input("Поиск по тексту", placeholder="Введите часть текста...", key="filter_search")
            is_annotated_option = st.radio(
                "Статус разметки",
                options=["Все", "Размечено", "Не размечено"],
                horizontal=True
            )
            is_annotated = None
            if is_annotated_option == "Размечено": is_annotated = True
            elif is_annotated_option == "Не размечено": is_annotated = False

        with col2:
            # Persistent filters from query params
            q_top5 = st.query_params.get_all("top5")
            q_human = st.query_params.get_all("human")
            
            # Initialize from query params if not yet in session state
            if "filter_top5" not in st.session_state:
                st.session_state.filter_top5 = [it for it in q_top5 if it in all_intents]
            if "filter_human" not in st.session_state:
                st.session_state.filter_human = [it for it in q_human if it in all_intents]
            
            top5_intents = st.multiselect(
                "Интенты (Top-5)",
                options=all_intents,
                key="filter_top5",
                help="Показать тексты, где эти интенты входят в Top-5 предсказаний"
            )
            human_intents = st.multiselect(
                "Интенты (аннотаторы)",
                options=all_intents,
                key="filter_human",
                help="Показать тексты, где аннотаторы выбрали эти интенты (Yes)"
            )
            
            # Sync back to query params
            if top5_intents != q_top5: st.query_params["top5"] = top5_intents
            if human_intents != q_human: st.query_params["human"] = human_intents
        
        with col3:
            assigned_annotators = st.multiselect(
                "Назначено на",
                options=all_annotators,
                key="filter_assigned",
                help="Фильтр по назначенным аннотаторам"
            )
            only_disagreements = st.toggle("Только разногласия", key="filter_disagreements", help="Показать тексты, где аннотаторы разошлись во мнениях")

    # 2. Base Count for pagination (needed before fetching data)
    total_count = stats_service.get_text_count(
        search_query=search_query,
        top5_intents=top5_intents,
        human_intents=human_intents,
        assigned_annotators=assigned_annotators,
        is_annotated=is_annotated,
        only_disagreements=only_disagreements
    )
    
    # Initial page settings
    # Initialize from query params if possible to persist across reloads
    if "text_overview_page" not in st.session_state:
        q_page = st.query_params.get("pg")
        st.session_state.text_overview_page = int(q_page) if q_page and q_page.isdigit() else 1
        
    if "text_overview_limit" not in st.session_state:
        q_limit = st.query_params.get("lim")
        if q_limit:
            st.session_state.text_overview_limit = int(q_limit) if q_limit.isdigit() else q_limit
        else:
            st.session_state.text_overview_limit = 100

    limit = st.session_state.text_overview_limit
    page_number = st.session_state.text_overview_page
    
    # Update query params to keep URL in sync
    st.query_params["pg"] = page_number
    st.query_params["lim"] = limit
    
    # Ensure page number is valid
    if limit != "Все":
        total_pages = (total_count - 1) // limit + 1
        if page_number > total_pages: 
            page_number = max(1, total_pages)
            st.session_state.text_overview_page = page_number
            st.query_params["pg"] = page_number
    else:
        page_number = 1
        limit_val = None
    
    limit_val = limit if limit != "Все" else None
    offset = (page_number - 1) * (limit_val or 0)

    # 3. Data Fetching
    df = stats_service.get_text_detailed_overview(
        search_query=search_query,
        top5_intents=top5_intents,
        human_intents=human_intents,
        assigned_annotators=assigned_annotators,
        is_annotated=is_annotated,
        only_disagreements=only_disagreements,
        limit=limit_val,
        offset=offset
    )
    
    if not df.empty:
        st.write(f"Показано текстов: **{len(df)}** из **{total_count}**")
        
        # Dynamic columns for human intents start with "Chosen Intent"
        chosen_cols = [c for c in df.columns if c.startswith("Chosen Intent")]
        model_cols = [f"Model Top {i}" for i in range(1, 6)]
        
        # Format disagreement column
        df['disagreement_icon'] = df['has_disagreement'].apply(lambda x: "⚠️" if x else "")
        
        # Display the dataframe
        st.dataframe(
            df,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "disagreement_icon": st.column_config.TextColumn("⚠️", width="small", help="Разногласия в разметке"),
                "text": st.column_config.TextColumn("Текст", width="large"),
                "assigned_to": "Назначено",
                "actual_annotators": "Размечали",
                "cluster": "Кластер",
                **{c: st.column_config.TextColumn(c, width="medium") for c in model_cols},
                **{c: st.column_config.TextColumn(c, width="medium") for c in chosen_cols}
            },
            column_order=["id", "disagreement_icon", "text", "assigned_to", "actual_annotators", "cluster"] + model_cols + chosen_cols,
            hide_index=True,
            width="stretch",
            height="content"  # Grow to fit content
        )
    else:
        st.info("Тексты не найдены")

    # 4. Pagination Section at the Bottom
    st.divider()
    b_col1, b_col2, b_col3 = st.columns([2, 2, 1])
    
    with b_col3:
        new_limit = st.selectbox(
            "Текстов на странице",
            options=[25, 50, 100, 250, "Все"],
            index=[25, 50, 100, 250, "Все"].index(limit) if limit in [25, 50, 100, 250, "Все"] else 2,
            key="text_overview_limit_select"
        )
        if new_limit != limit:
            st.session_state.text_overview_limit = new_limit
            st.session_state.text_overview_page = 1
            st.rerun()

    if limit != "Все":
        total_pages = (total_count - 1) // limit + 1
        if total_pages > 1:
            with b_col1:
                # Optimized pagination layout: [First][Prev] [Numbers + Dropdown] [Next][Last]
                pg_cols = st.columns([0.6, 0.6, 5.0, 0.6, 0.6])
                
                with pg_cols[0]:
                    if st.button("⏪", disabled=(page_number == 1), key="pg_first", use_container_width=True):
                        st.session_state.text_overview_page = 1
                        st.session_state.pg_jump_select = 1
                        st.rerun()
                
                with pg_cols[1]:
                    if st.button("◀️", disabled=(page_number == 1), key="pg_prev", use_container_width=True):
                        new_pg = max(1, page_number - 1)
                        st.session_state.text_overview_page = new_pg
                        st.session_state.pg_jump_select = new_pg
                        st.rerun()
                
                with pg_cols[2]:
                    # Vertical stack: Numbers on top, Dropdown below
                    # 1. Page numbers
                    start_pg = max(1, page_number - 2)
                    end_pg = min(total_pages, page_number + 2)
                    if end_pg - start_pg < 4:
                        if start_pg == 1: end_pg = min(total_pages, 5)
                        elif end_pg == total_pages: start_pg = max(1, total_pages - 4)
                    
                    p_nums = list(range(start_pg, end_pg + 1))
                    pg_nums = st.columns(len(p_nums))
                    for idx, p in enumerate(p_nums):
                        if pg_nums[idx].button(
                            str(p), 
                            type="primary" if p == page_number else "secondary",
                            key=f"pg_num_{p}",
                            use_container_width=True
                        ):
                            st.session_state.text_overview_page = p
                            st.session_state.pg_jump_select = p
                            st.rerun()
                    
                    # 2. Jump to page dropdown (compact version)
                    if "pg_jump_select" not in st.session_state or st.session_state.pg_jump_select != page_number:
                        st.session_state.pg_jump_select = page_number
                        
                    chosen_page = st.selectbox(
                        "К стр.",
                        options=list(range(1, total_pages + 1)),
                        key="pg_jump_select",
                        label_visibility="collapsed"
                    )
                    if chosen_page != page_number:
                        st.session_state.text_overview_page = chosen_page
                        st.rerun()

                with pg_cols[3]:
                    if st.button("▶️", disabled=(page_number == total_pages), key="pg_next", use_container_width=True):
                        new_pg = min(total_pages, page_number + 1)
                        st.session_state.text_overview_page = new_pg
                        st.session_state.pg_jump_select = new_pg
                        st.rerun()
                        
                with pg_cols[4]:
                    if st.button("⏩", disabled=(page_number == total_pages), key="pg_last", use_container_width=True):
                        st.session_state.text_overview_page = total_pages
                        st.session_state.pg_jump_select = total_pages
                        st.rerun()
    
    with b_col2:
        st.write(f"Всего текстов: {total_count}")


def show_cluster_progress(stats_service: StatsService):
    """Display cluster progress."""
    st.header("📈 Прогресс по кластерам")
    
    df = stats_service.get_cluster_progress()
    
    if not df.empty:
        # Format completion rate
        df['completion_rate'] = df['completion_rate'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(
            df,
            column_config={
                "cluster": "Кластер",
                "total_texts": "Всего текстов",
                "annotated_texts": "Размечено",
                "completion_rate": "% Завершено"
            },
            hide_index=True,
            width="stretch"
        )
        
        # Visualize progress
        st.bar_chart(df.set_index('cluster')['annotated_texts'])
    else:
        st.info("Нет данных о кластерах")


def show_disagreements(stats_service: StatsService):
    """Display annotation disagreements."""
    st.header("⚠️ Разногласия")
    
    df = stats_service.get_disagreements(min_annotators=settings.min_annotators)
    
    if not df.empty:
        st.write(f"Найдено {len(df)} случаев разногласий:")
        
        st.dataframe(
            df,
            column_config={
                "text_id": "ID текста",
                "request_text": st.column_config.TextColumn("Текст", width="large"),
                "label": "Интент",
                "annotator_count": "Аннотаторов",
                "yes_count": "Да",
                "no_count": "Нет"
            },
            hide_index=True,
            width="stretch"
        )
    else:
        st.success("✅ Разногласий не найдено")


def show_export_section(stats_service: StatsService):
    """Display export options."""
    st.header("💾 Экспорт данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Экспортировать аннотации", type="primary"):
            output_path = f"exports/annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs("exports", exist_ok=True)
            
            count = stats_service.export_annotations(output_path)
            st.success(f"✅ Экспортировано {count} записей в {output_path}")


def show_import_section():
    """Display import options with file upload."""
    st.header("📥 Импорт данных")
    
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Загрузить файл",
        type=["csv"],
        help="Выберите CSV файл для импорта текстов"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = uploads_dir / f"upload_{timestamp}.csv"
        
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Файл сохранен: {upload_path}")
        
        # Validate and preview CSV
        try:
            # Read CSV
            df = pd.read_csv(upload_path)
            
            # Validate required columns
            required_columns = ["request_text"]
            optional_columns = ["language", "clusters"]
            missing_required = [col for col in required_columns if col not in df.columns]
            
            if missing_required:
                st.error(f"❌ Отсутствуют обязательные колонки: {', '.join(missing_required)}")
                st.info("Требуемые колонки: request_text")
                st.info("Опциональные колонки: language, clusters, score_<intent_name>")
                return
            
            # Show file info
            st.subheader("📋 Информация о файле")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Всего строк", len(df))
            
            with col2:
                non_empty = df["request_text"].notna().sum()
                st.metric("Непустых текстов", non_empty)
            
            # Show detected columns
            st.subheader("🔍 Обнаруженные колонки")
            
            detected_required = [col for col in required_columns if col in df.columns]
            detected_optional = [col for col in optional_columns if col in df.columns]
            score_columns = [col for col in df.columns if col.startswith("score_")]
            
            if detected_required:
                st.success(f"✅ Обязательные: {', '.join(detected_required)}")
            
            if detected_optional:
                st.info(f"ℹ️ Опциональные: {', '.join(detected_optional)}")
            
            if score_columns:
                st.info(f"📊 Оценки интентов: {', '.join(score_columns)}")
            
            # Show preview
            st.subheader("👁️ Предварительный просмотр")
            st.dataframe(df.head(10), width="stretch")
            
            # Import button
            st.subheader("▶️ Выполнить импорт")
            
            if st.button("🚀 Импортировать данные", type="primary"):
                with st.spinner("Импортирую данные..."):
                    try:
                        # Get current versions
                        with get_connection(settings.db_path) as conn:
                            model_version_row = conn.execute(
                                "SELECT value FROM settings WHERE key = 'current_model_version'"
                            ).fetchone()
                            data_version_row = conn.execute(
                                "SELECT value FROM settings WHERE key = 'current_data_version'"
                            ).fetchone()
                            
                            model_version = int(model_version_row["value"]) if model_version_row else 0
                            data_version = int(data_version_row["value"]) if data_version_row else 0
                        
                        # Load intents
                        intent_repo = IntentRepository(settings.db_path)
                        intents = intent_repo.load_from_yaml(settings.intents_path)
                        
                        # Import from CSV
                        import_service = ImportService(settings.db_path)
                        
                        # Track detailed stats
                        total_rows = len(df)
                        imported_count = import_service.import_from_csv(
                            csv_path=str(upload_path),
                            intents=intents,
                            top_k=settings.top_k,
                            model_version=model_version,
                            data_version=data_version
                        )
                        
                        skipped_count = total_rows - imported_count
                        
                        # Show detailed report
                        st.success("✅ Импорт завершен")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Всего строк", total_rows)
                        
                        with col2:
                            st.metric("Импортировано", imported_count, delta=imported_count, delta_color="normal")
                        
                        with col3:
                            st.metric("Пропущено (дубликаты)", skipped_count, delta=-skipped_count if skipped_count > 0 else 0)
                        
                        st.info(f"📁 Файл сохранен: {upload_path}")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка импорта: {str(e)}")
                        
        except pd.errors.EmptyDataError:
            st.error("❌ Файл пустой или поврежден")
        except pd.errors.ParserError as e:
            st.error(f"❌ Ошибка парсинга CSV: {str(e)}")
            st.info("Убедитесь, что файл имеет корректный формат CSV")
        except Exception as e:
            st.error(f"❌ Ошибка чтения файла: {str(e)}")
    else:
        st.info("👆 Выберите CSV файл для импорта")
        
        # Show requirements
        with st.expander("📖 Требования к формату файла"):
            st.markdown("""
            **Обязательные колонки:**
            - `request_text` - текст для импорта
            
            **Опциональные колонки:**
            - `language` - язык текста (например: ru, en)
            - `clusters` - список кластеров через запятую
            - `score_<intent_name>` - оценки для интентов (например: score_greeting, score_question)
            
            **Пример CSV:**
            ```
            request_text,language,clusters,score_greeting,score_question
            "Привет, как дела?",ru,"general,greetings",0.9,0.1
            "Что это?",ru,"general,questions",0.1,0.8
            ```
            """)



def main():
    """Main admin dashboard."""
    st.title("📊 Панель администратора")
    
    # Authentication
    authenticate_admin()
    
    # Initialize service
    stats_service = StatsService(settings.db_path)
    
    # Navigation Tabs Definition
    tabs = [
        "📊 Обзор",
        "👥 Аннотаторы",
        "🎯 Качество",
        "📝 Тексты",
        "📈 Кластеры",
        "⚠️ Разногласия",
        "💾 Экспорт",
        "📥 Импорт"
    ]
    
    # Persistent tab via cookies
    saved_tab = cookie_manager.get("admin_active_tab")
    q_tab = st.query_params.get("tab")
    
    # Initialize session state if fresh
    if "admin_active_tab" not in st.session_state:
        if q_tab in tabs:
            st.session_state.admin_active_tab = q_tab
        elif saved_tab in tabs:
            st.session_state.admin_active_tab = saved_tab
        else:
            st.session_state.admin_active_tab = tabs[0]
        st.session_state.last_tab_param = st.session_state.admin_active_tab
    
    # Sync from URL if it changed externally (e.g. browser back button)
    if q_tab and q_tab in tabs and q_tab != st.session_state.get("last_tab_param"):
        st.session_state.admin_active_tab = q_tab
        st.session_state.last_tab_param = q_tab
        # Also sync filters from URL if we just switched tab via URL
        if "human" in st.query_params:
            st.session_state.filter_human = st.query_params.get_all("human")
        if "top5" in st.query_params:
            st.session_state.filter_top5 = st.query_params.get_all("top5")

    selected_tab = st.pills(
        "Навигация",
        options=tabs,
        key="admin_active_tab",
        label_visibility="collapsed"
    )
    
    # Update persistent state
    if selected_tab:
        if selected_tab != saved_tab:
            cookie_manager.set("admin_active_tab", selected_tab, expires_at=datetime.now() + timedelta(days=30))
        
        # Keep URL in sync
        if st.query_params.get("tab") != selected_tab:
            st.query_params["tab"] = selected_tab
            st.session_state.last_tab_param = selected_tab
    
    st.divider()

    if selected_tab == "📊 Обзор":
        show_overall_stats(stats_service)
    
    elif selected_tab == "👥 Аннотаторы":
        show_annotator_stats(stats_service)
    
    elif selected_tab == "🎯 Качество":
        show_intent_quality(stats_service)
    
    elif selected_tab == "📝 Тексты":
        show_text_overview(stats_service)
    
    elif selected_tab == "📈 Кластеры":
        show_cluster_progress(stats_service)
    
    elif selected_tab == "⚠️ Разногласия":
        show_disagreements(stats_service)
    
    elif selected_tab == "💾 Экспорт":
        show_export_section(stats_service)
    
    elif selected_tab == "📥 Импорт":
        show_import_section()



if __name__ == "__main__":
    main()
