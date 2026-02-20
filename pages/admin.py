"""Admin Dashboard - Refactored to use StatsService."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import extra_streamlit_components as stx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, set_key

load_dotenv(Path(__file__).parent.parent / ".env")


from src.repositories.intent_repo import IntentRepository
from src.services.annotation_service import AnnotationService
from src.services.auth_service import AuthService
from src.services.import_service import ImportService
from src.services.stats_service import StatsService
from src.utils.config import settings, LANGUAGE_NAMES
from src.utils.database import get_connection, init_database

# Admin password
ADMIN_PASSWORD = os.environ.get("TEXTS_ADMIN_PASSWORD", "admin123")

# Read layout preference from query params
_layout_pref = st.query_params.get("layout", "wide")
if _layout_pref not in ("wide", "centered"):
    _layout_pref = "wide"

st.set_page_config(page_title="Admin Dashboard", layout=_layout_pref, page_icon="📊")

# Dark theme CSS
DARK_THEME_CSS = """
<style>
    :root {
        --primary-color: #ff4b4b;
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #fafafa;
        color-scheme: dark;
    }
    .stApp, [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; }
    [data-testid="stHeader"] { background-color: #0e1117; }
    [data-testid="stSidebar"] [data-testid="stMarkdown"],
    [data-testid="stSidebar"] label { color: #fafafa; }
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] { background-color: #262730; }
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #262730; color: #fafafa; border-color: #4a4a5a;
    }
    hr { border-color: #4a4a5a; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #fafafa; }
    [data-testid="stExpander"] { border-color: #4a4a5a; }
    .stDataFrame, .stTable { color: #fafafa; }
    div[data-baseweb="popover"] ul { background-color: #262730; }
    div[data-baseweb="popover"] li:hover { background-color: #3a3a4a; }
</style>
"""

if st.query_params.get("dark") == "1":
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

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
        total_texts = int(row['total_texts'])
        annotated_texts = int(row['fully_annotated_texts'])
        pending_texts = total_texts - annotated_texts
        extra_intents_texts = int(row['texts_with_extra_intents'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Всего текстов", f"{total_texts:,}")
        
        with col2:
            st.metric("Текстов размечено", f"{annotated_texts:,}")
        
        with col3:
            st.metric("Осталось разметить", f"{pending_texts:,}")
        
        with col4:
            if total_texts > 0:
                extra_pct = (extra_intents_texts / total_texts) * 100
                st.metric("% c доп. интентами", f"{extra_pct:.1f}%")
            else:
                st.metric("% c доп. интентами", "0.0%")

    # Divider
    st.divider()

    # Intent Model Quality Block (Restored)
    st.header("Качество модели по интентам")
    st.markdown("""
    Метрики для понимания, какие интенты модель предсказывает хорошо, а какие — плохо:
    - **Top-1 Precision** — когда интент на 1 месте, как часто разметчик ставит "yes"
    - **Missed Rate** — когда интент на 2-N месте И top-1 отвергнут (no), как часто этот интент получает "yes"
      (высокий missed rate = модель часто ставит этот интент ниже, чем нужно)
    """)

    model_quality = stats_service.get_model_quality_legacy()
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
                         "missed_opportunity_count", "missed_rate"]],
            column_config={
                "label": "Интент",
                "cluster": "Кластер",
                "top1_count": "Top-1 раз",
                "top1_yes": "Top-1 Yes",
                "top1_no": "Top-1 No",
                "top1_precision": st.column_config.NumberColumn("Top-1 Precision", format="%.1f%%"),
                "potential_count": "Потенц. пропусков",
                "missed_opportunity_count": "Факт. пропусков",
                "missed_rate": st.column_config.NumberColumn("Missed Rate", format="%.1f%%")
            },
            width="stretch",
            hide_index=True
        )

        # Highlight problematic intents
        st.subheader("Интенты, требующие внимания")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Низкий Top-1 Precision**")
            st.caption("Модель уверена, но ошибается")
            low_top1_prec = model_quality[model_quality["top1_count"] >= 3].nsmallest(10, "top1_precision")
            
            if not low_top1_prec.empty:
                # Calculate Disagreement Rate
                low_top1_prec["disagreement_rate"] = low_top1_prec.apply(
                    lambda r: r["top1_no"] / r["top1_count"] if r["top1_count"] > 0 else 0.0,
                    axis=1
                )
                
                st.dataframe(
                    low_top1_prec[["label", "top1_precision", "top1_count", "disagreement_rate"]],
                    column_config={
                        "label": "Интент",
                        "top1_precision": st.column_config.NumberColumn("Precision", format="%.1f%%"),
                        "top1_count": "Top-1 раз",
                        "disagreement_rate": st.column_config.NumberColumn("Disagreement Rate", format="%.1f%%")
                    },
                    width="stretch",
                    hide_index=True
                )

        with col2:
            st.markdown("**Высокий Missed Rate**")
            st.caption("Модель недооценивает этот интент")
            
            # Metric Explanation
            with st.expander("ℹ️ Что означают эти метрики?"):
                st.markdown("""
                - **Пропущено (Missed Opportunity)**: Количество раз, когда этот интент был правильным (Yes), но модель поставила его **не на 1 место**.
                - **Потенциал (Potential Count)**: Количество раз, когда этот интент был кандидатом (2-5 место), а Top-1 был отвергнут. Это ситуации, где модель могла бы "спасти" аннотацию, если бы ранжировала лучше.
                - **Missed Rate**: Пропущено / Потенциал. Высокий процент означает, что модель часто ставит верный интент на вторые роли.
                """)
            
            high_missed = model_quality[model_quality["potential_count"] >= 3].nlargest(10, "missed_rate")
            if not high_missed.empty:
                st.dataframe(
                    high_missed[["label", "missed_rate", "missed_opportunity_count", "potential_count"]],
                    column_config={
                        "label": "Интент",
                        "missed_rate": st.column_config.NumberColumn("Missed Rate", format="%.1f%%"),
                        "missed_opportunity_count": "Пропущено",
                        "potential_count": "Потенциал"
                    },
                    width="stretch",
                    hide_index=True
                )
    else:
        st.info("Недостаточно данных для расчета метрик качества модели.")





def show_annotator_stats(stats_service: StatsService):
    """Display annotator statistics."""
    st.header("👥 Статистика разметчиков")
    
    # 1. General Stats Table
    df = stats_service.get_annotator_stats()
    
    if not df.empty:
        # Format percentages
        df['yes_rate'] = df['yes_rate'].apply(lambda x: f"{x*100:.1f}%")
        df['annotated_pct'] = df['annotated_pct'].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(
            df,
            column_config={
                "annotator": "Разметчик",
                "texts_assigned": "Назначено текстов",
                "texts_annotated": "Размечено текстов",
                "annotated_pct": "% Размечено",
                "total_decisions": "Всего решений",
                "yes_count": "Да",
                "no_count": "Нет",
                "yes_rate": "% Да"
            },
            column_order=[
                "annotator", "texts_assigned", "texts_annotated", "annotated_pct",
                "total_decisions", "yes_count", "no_count", "yes_rate"
            ],
            hide_index=True,
            width="stretch"
        )
    else:
        st.info("Нет данных")

    st.divider()

    # 2. Activity Charts (Moved from Activity Stats)
    # Daily Activity
    st.subheader("Активность по дням")
    daily = stats_service.get_daily_activity()
    if not daily.empty:
        chart = alt.Chart(daily).mark_bar().encode(
            x=alt.X('date:O', axis=alt.Axis(labelAngle=0, title='Дата')),
            y=alt.Y('count:Q', title='Количество'),
            color=alt.Color('annotator:N', title='Разметчик'),
            tooltip=['date', 'annotator', 'count']
        ).interactive()
        st.altair_chart(chart, width="stretch")
    
    # Hourly Activity
    st.subheader("Активность по часам")
    hourly = stats_service.get_hourly_activity()
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
            
            # Use Melted data for Altair to handle stacked bars easily or just use the filtered data directly
            # To ensure we show the breakdown by annotator, we use the original 'hourly_filtered'
            
            chart = alt.Chart(hourly_filtered).mark_bar().encode(
                x=alt.X('hour:O', axis=alt.Axis(labelAngle=0, title='Час')),
                y=alt.Y('count:Q', title='Количество'),
                color=alt.Color('annotator:N', title='Разметчик'),
                tooltip=['hour', 'annotator', 'count']
            ).interactive()
            st.altair_chart(chart, width="stretch")
            
            # Summary table
            summary = hourly_filtered.groupby("annotator").agg(
                total=("count", "sum"),
                hours_active=("hour", "nunique"),
                first_hour=("hour", "min"),
                last_hour=("hour", "max")
            ).reset_index()
            summary.columns = ["Разметчик", "Всего", "Часов", "Начало", "Конец"]
            st.dataframe(summary, width="stretch", hide_index=True)
    else:
        st.info("Нет данных об активности по часам.")


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
            width="stretch",
            height="content"
        )

    else:
        st.info("Нет данных о качестве интентов")


def show_text_overview(stats_service: StatsService):
    """Display detailed text overview with filters and pagination."""
    st.header("📝 Обзор текстов")
    
    # All metadata for filters
    all_intents = stats_service.get_all_intents()
    all_annotators = stats_service.get_unique_assigned_annotators()
    all_languages = stats_service.get_unique_languages()

    # — Assign unassigned button —
    with st.expander("⚡ Действия", expanded=False):
        st.markdown(
            "**Назначить неназначенные тексты** — запускает алгоритм назначения "
            "для всех текстов без разметчика (или с устаревшим назначением)."
        )
        if st.button("🔄 Назначить неназначенные", key="btn_assign_unassigned"):
            with st.spinner("Назначаю тексты..."):
                try:
                    auth_service = AuthService(settings.annotators_path)
                    annotators = auth_service.load_annotators().annotators
                    annotation_service = AnnotationService(settings.db_path)
                    updated = annotation_service.assign_unannotated_texts(annotators)
                    if updated:
                        st.success(f"✅ Назначено / переназначено текстов: **{updated}**")
                    else:
                        st.info("ℹ️ Все тексты уже корректно назначены, изменений нет.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")

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
            if all_languages:
                selected_languages = st.multiselect(
                    "Язык",
                    options=all_languages,
                    key="filter_language",
                    help="Фильтр по языку текста",
                    format_func=lambda code: LANGUAGE_NAMES.get(code, code)
                )
            else:
                selected_languages = []

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
                "Интенты (разметчики)",
                options=all_intents,
                key="filter_human",
                help="Показать тексты, где разметчики выбрали эти интенты (Yes)"
            )
            
            # Sync back to query params
            if top5_intents != q_top5: st.query_params["top5"] = top5_intents
            if human_intents != q_human: st.query_params["human"] = human_intents
        
        with col3:
            q_assigned = st.query_params.get_all("assigned")
            assigned_options = ["[Unassigned]"] + all_annotators
            if "filter_assigned" not in st.session_state and q_assigned:
                st.session_state.filter_assigned = [a for a in q_assigned if a in assigned_options]
            assigned_annotators = st.multiselect(
                "Назначено на",
                options=assigned_options,
                key="filter_assigned",
                help="Фильтр по назначенным разметчикам (включая [Unassigned] для текстов без назначения)"
            )
            only_disagreements = st.toggle("Только разногласия", key="filter_disagreements", help="Показать тексты, где разметчики разошлись во мнениях")

    # 2. Base Count for pagination (needed before fetching data)
    total_count = stats_service.get_text_count(
        search_query=search_query,
        top5_intents=top5_intents,
        human_intents=human_intents,
        assigned_annotators=assigned_annotators,
        languages=selected_languages if selected_languages else None,
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
        languages=selected_languages if selected_languages else None,
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
                    if st.button("⏪", disabled=(page_number == 1), key="pg_first", width="stretch"):
                        st.session_state.text_overview_page = 1
                        st.session_state.pg_jump_select = 1
                        st.rerun()
                
                with pg_cols[1]:
                    if st.button("◀️", disabled=(page_number == 1), key="pg_prev", width="stretch"):
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
                            width="stretch"
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
                    if st.button("▶️", disabled=(page_number == total_pages), key="pg_next", width="stretch"):
                        new_pg = min(total_pages, page_number + 1)
                        st.session_state.text_overview_page = new_pg
                        st.session_state.pg_jump_select = new_pg
                        st.rerun()
                        
                with pg_cols[4]:
                    if st.button("⏩", disabled=(page_number == total_pages), key="pg_last", width="stretch"):
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
        # Visualize progress
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('cluster:N', axis=alt.Axis(labelAngle=0, title='Кластер')),
            y=alt.Y('annotated_texts:Q', title='Размечено'),
            tooltip=['cluster', 'total_texts', 'annotated_texts', 'completion_rate']
        ).interactive()
        st.altair_chart(chart, width="stretch")
    else:
        st.info("Нет данных о кластерах")


def show_disagreements(stats_service: StatsService):
    """Display annotation disagreements."""
    st.header("⚠️ Разногласия")
    
    df = stats_service.get_disagreements()
    
    if not df.empty:
        st.write(f"Найдено {len(df)} случаев разногласий:")
        
        st.dataframe(
            df,
            column_config={
                "text_id": "ID текста",
                "request_text": st.column_config.TextColumn("Текст", width="large"),
                "label": "Интент",
                "annotator_count": "Разметчиков",
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
    
    col1 = st.columns(1)[0]
    
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
                        
                        # Load intents and sync to DB (so Texts tab filters work)
                        intent_repo = IntentRepository(settings.db_path)
                        intents = intent_repo.load_from_yaml(settings.intents_path)
                        for intent in intents.values():
                            intent_repo.upsert(intent)

                        # Load annotators for assignment
                        auth_service = AuthService(settings.annotators_path)
                        annotators = auth_service.load_annotators().annotators
                        annotation_service = AnnotationService(settings.db_path)

                        # Import from CSV
                        import_service = ImportService(settings.db_path)

                        # Track detailed stats
                        total_rows = len(df)
                        imported_count = import_service.import_from_csv(
                            csv_path=str(upload_path),
                            intents=intents,
                            top_k=settings.top_k,
                            model_version=model_version,
                            data_version=data_version,
                            probability_threshold=settings.probability_threshold,
                            annotators=annotators,
                            annotation_service=annotation_service
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

                        # Check for unassigned texts
                        with get_connection(settings.db_path) as conn:
                            unassigned = conn.execute(
                                "SELECT COUNT(*) FROM texts WHERE assigned_to IS NULL OR assigned_to = ''"
                            ).fetchone()[0]
                        if unassigned > 0:
                            texts_tab = "📝 Тексты"
                            link = f"/admin?tab={texts_tab}&assigned=[Unassigned]"
                            st.warning(
                                f"⚠️ {unassigned} текстов не назначены разметчикам. "
                                f"[Открыть в Текстах]({link})"
                            )

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

    # Ensure DB schema exists (in case app.py hasn't run yet)
    init_database(settings.db_path)

    # Authentication
    authenticate_admin()

    # Initialize service
    stats_service = StatsService(settings.db_path)
    
    # Navigation Tabs Definition
    tabs = [
        "📊 Обзор",
        "👥 Разметчики",
        "🎯 Качество",
        "📝 Тексты",
        "📈 Кластеры",
        "💾 Экспорт",
        "📥 Импорт",
        "⚙️ Настройки"
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
    
    elif selected_tab == "👥 Разметчики":
        show_annotator_stats(stats_service)
    
    elif selected_tab == "🎯 Качество":
        show_intent_quality(stats_service)
    
    elif selected_tab == "📝 Тексты":
        show_text_overview(stats_service)
    
    elif selected_tab == "📈 Кластеры":
        show_cluster_progress(stats_service)
    
    elif selected_tab == "💾 Экспорт":
        show_export_section(stats_service)
    
    elif selected_tab == "📥 Импорт":
        show_import_section()

    elif selected_tab == "⚙️ Настройки":
        show_settings(stats_service)


_ENV_PATH = str(Path(__file__).parent.parent / ".env")

# Mapping: (env_key, settings_attr_name)
_ENV_FIELDS = {
    "set_dump_interval": ("TEXTS_DB_DUMP_INTERVAL_SEC", "db_dump_interval"),
    "set_top_k": ("TEXTS_TOP_K", "top_k"),
    "set_prob_threshold": ("TEXTS_PROBABILITY_THRESHOLD", "probability_threshold"),
    "set_conf_threshold": ("ANNOTATORS_INTENTS_CONFIDENCE_THRESHOLD", "annotators_intents_confidence_threshold"),
    "set_margin_threshold": ("TEXTS_MARGIN_THRESHOLD", "margin_threshold"),
    "set_log_level": ("LOG_LEVEL", "log_level"),
}


_PERCENT_FIELDS = {"set_prob_threshold", "set_conf_threshold", "set_margin_threshold"}


def _save_settings_to_env():
    """Write current widget values to .env and update in-memory settings."""
    for widget_key, (env_key, attr_name) in _ENV_FIELDS.items():
        value = st.session_state.get(widget_key)
        if value is not None:
            if attr_name in ("db_dump_interval", "top_k"):
                value = int(value)
            elif widget_key in _PERCENT_FIELDS:
                value = round(value / 100.0, 4)
            set_key(_ENV_PATH, env_key, str(value))
            setattr(settings, attr_name, value)


def _load_settings_from_env():
    """Re-read .env into a fresh Settings instance and update global settings."""
    from src.utils.config import Settings
    fresh = Settings()
    for _, (_, attr_name) in _ENV_FIELDS.items():
        setattr(settings, attr_name, getattr(fresh, attr_name))


def show_settings(stats_service: StatsService):
    """Settings tab for application configuration."""
    st.subheader("Настройки приложения")

    # --- UI Settings (stored in DB) ---
    st.markdown("#### Интерфейс разметки")

    current_conf = stats_service.get_setting("show_confidence", "0")
    show_conf = st.checkbox(
        "Показывать уверенность модели рядом с интентами",
        value=current_conf == "1",
        help="Если включено, рядом с каждым интентом будет отображаться процент уверенности модели"
    )
    new_conf = "1" if show_conf else "0"
    if new_conf != current_conf:
        stats_service.set_setting("show_confidence", new_conf)
        st.rerun()

    # --- .env parameters ---
    st.divider()
    st.markdown("#### Параметры приложения")

    # -- Database --
    st.markdown("**База данных**")
    st.number_input(
        "Интервал автосохранения дампа (сек)",
        value=settings.db_dump_interval, min_value=10, step=10,
        help="TEXTS_DB_DUMP_INTERVAL_SEC — минимум 10",
        key="set_dump_interval",
    )

    # -- Application --
    st.markdown("**Модель и пороги**")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Количество показываемых интентов (Top-K)",
            value=settings.top_k, min_value=1, max_value=20, step=1,
            help="TEXTS_TOP_K — от 1 до 20",
            key="set_top_k",
        )
        st.number_input(
            "Порог вероятности для импорта кандидатов, %",
            value=int(settings.probability_threshold * 100), min_value=0, max_value=100, step=5,
            help="TEXTS_PROBABILITY_THRESHOLD — кандидаты ниже порога не импортируются (0–100%)",
            key="set_prob_threshold",
        )

    with col2:
        st.number_input(
            "Порог уверенности для интентов разметчика, %",
            value=int(settings.annotators_intents_confidence_threshold * 100),
            min_value=0, max_value=100, step=5,
            help="ANNOTATORS_INTENTS_CONFIDENCE_THRESHOLD — минимальная уверенность для показа интента разметчику (0–100%)",
            key="set_conf_threshold",
        )
        st.number_input(
            "Порог маржи между кандидатами, %",
            value=int(settings.margin_threshold * 100), min_value=0, max_value=100, step=5,
            help="TEXTS_MARGIN_THRESHOLD — минимальная разница вероятностей между кандидатами (0–100%)",
            key="set_margin_threshold",
        )

    # -- Logging --
    st.markdown("**Логирование**")
    st.selectbox(
        "Уровень логирования",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
        index=["DEBUG", "INFO", "WARNING", "ERROR"].index(settings.log_level),
        help="LOG_LEVEL",
        key="set_log_level",
    )

    # -- Save / Load buttons --
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Сохранить в .env"):
            _save_settings_to_env()
            st.success("Сохранено в .env")
    with c2:
        if st.button("📂 Загрузить из .env"):
            _load_settings_from_env()
            st.rerun()


if __name__ == "__main__":
    main()
