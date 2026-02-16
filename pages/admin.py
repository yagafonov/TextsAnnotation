"""Admin Dashboard - Refactored to use StatsService."""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from src.repositories.base import BaseRepository
from src.repositories.intent_repo import IntentRepository
from src.services.import_service import ImportService
from src.services.stats_service import StatsService
from src.utils.config import settings
from src.utils.database import get_connection

# Admin password
ADMIN_PASSWORD = os.environ.get("TEXTS_ADMIN_PASSWORD", "admin123")

st.set_page_config(page_title="Admin Dashboard", layout="wide", page_icon="📊")


def authenticate_admin():
    """Admin authentication with session management."""
    # Initialize session state
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    # Show logout button if authenticated
    if st.session_state.admin_authenticated:
        with st.sidebar:
            st.success("✅ Администратор авторизован")
            if st.button("🚪 Выйти", use_container_width=True):
                st.session_state.admin_authenticated = False
                st.rerun()
        return
    
    # Show login form
    with st.sidebar:
        st.header("🔐 Авторизация")
        
        admin_password = st.text_input("Пароль администратора", type="password")
        if st.button("Войти", use_container_width=True):
            if admin_password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.success("✅ Вход выполнен")
                st.rerun()
            else:
                st.error("❌ Неверный пароль")
    
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
            use_container_width=True
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
                "miss_rate": "% Пропусков"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Нет данных о качестве интентов")


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
            use_container_width=True
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
                "text": st.column_config.TextColumn("Текст", width="large"),
                "label": "Интент",
                "annotator_count": "Аннотаторов",
                "yes_count": "Да",
                "no_count": "Нет"
            },
            hide_index=True,
            use_container_width=True
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
    
    with col2:
        # Create data/model versions (matching original db.py behavior)
        repo = BaseRepository(settings.db_path)
        
        if st.button("🏷️ Создать версию данных"):
            with get_connection(settings.db_path) as conn:
                # Get current version from settings
                result = conn.execute("SELECT value FROM settings WHERE key = 'current_data_version'").fetchone()
                current = int(result["value"]) if result else 0
                new_version = current + 1
                
                # Update setting only (original behavior - no table insert)
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                    ("current_data_version", str(new_version))
                )
                conn.commit()
            st.success(f"✅ Создана версия данных #{new_version}")
        
        if st.button("🤖 Создать версию модели"):
            with get_connection(settings.db_path) as conn:
                # Get current version from settings
                result = conn.execute("SELECT value FROM settings WHERE key = 'current_model_version'").fetchone()
                current = int(result["value"]) if result else 0
                new_version = current + 1
                
                # Insert into model_versions table AND update setting (original behavior)
                conn.execute(
                    "INSERT INTO model_versions (version, note, created_at) VALUES (?, ?, datetime('now'))",
                    (new_version, "Manual version")
                )
                conn.execute(
                    "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                    ("current_model_version", str(new_version))
                )
                conn.commit()
            st.success(f"✅ Создана версия модели #{new_version}")


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
            required_columns = ["text"]
            optional_columns = ["language", "clusters"]
            missing_required = [col for col in required_columns if col not in df.columns]
            
            if missing_required:
                st.error(f"❌ Отсутствуют обязательные колонки: {', '.join(missing_required)}")
                st.info("Требуемые колонки: text")
                st.info("Опциональные колонки: language, clusters, score_<intent_name>")
                return
            
            # Show file info
            st.subheader("📋 Информация о файле")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Всего строк", len(df))
            
            with col2:
                non_empty = df["text"].notna().sum()
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
            st.dataframe(df.head(10), use_container_width=True)
            
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
            - `text` - текст для импорта
            
            **Опциональные колонки:**
            - `language` - язык текста (например: ru, en)
            - `clusters` - список кластеров через запятую
            - `score_<intent_name>` - оценки для интентов (например: score_greeting, score_question)
            
            **Пример CSV:**
            ```
            text,language,clusters,score_greeting,score_question
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
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Обзор",
        "👥 Аннотаторы",
        "🎯 Качество",
        "📈 Кластеры",
        "⚠️ Разногласия",
        "💾 Экспорт",
        "📥 Импорт"
    ])
    
    with tab1:
        show_overall_stats(stats_service)
    
    with tab2:
        show_annotator_stats(stats_service)
    
    with tab3:
        show_intent_quality(stats_service)
    
    with tab4:
        show_cluster_progress(stats_service)
    
    with tab5:
        show_disagreements(stats_service)
    
    with tab6:
        show_export_section(stats_service)
    
    with tab7:
        show_import_section()



if __name__ == "__main__":
    main()
