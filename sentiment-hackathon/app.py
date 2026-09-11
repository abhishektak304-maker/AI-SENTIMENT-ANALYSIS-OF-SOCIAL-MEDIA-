"""SocialPulse: AI Sentiment & Topic Intelligence Dashboard.

A responsive, production-grade analytical dashboard for social media sentiment,
emotion, topic, and trend analysis supporting English and Hinglish.
Features comprehensive theme awareness (Dark & Light), collapsible sidebar,
responsive grids, and zero-overflow layout.
"""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    analyze_dataframe,
    analyze_single_post,
    detect_sentiment_spikes,
    get_emotion_summary,
    get_hashtag_trends,
    get_keyword_trends,
    get_sentiment_over_time,
    get_sentiment_summary,
    get_topic_summary,
)
from src.utils import (
    ActionRateLimiter,
    sanitize_input_text,
    validate_uploaded_file,
)
from src.database import (
    init_db,
    get_all_posts,
    insert_post,
    insert_posts_batch,
    get_db_stats,
    reset_db,
)

# Application Page Configuration
st.set_page_config(
    page_title="SocialPulse | Sentiment & Topic Intelligence",
    page_icon="favicon.png" if os.path.exists("favicon.png") else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "SocialPulse: Social Media Sentiment and Topic Intelligence Dashboard."
    }
)

# Initialize Session State
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "current_view" not in st.session_state:
    st.session_state.current_view = "Analytics Overview"

if "settings_open" not in st.session_state:
    st.session_state.settings_open = False

if "sim_active" not in st.session_state:
    st.session_state.sim_active = False

if "sim_last_post" not in st.session_state:
    st.session_state.sim_last_post = None

if "rate_limiter" not in st.session_state:
    st.session_state.rate_limiter = ActionRateLimiter(max_requests=50, window_seconds=60.0)


@st.cache_data
def load_bundled_data() -> pd.DataFrame:
    """Check the SQLite database; if not present create and seed one, then load."""
    init_ok, init_msg = init_db()
    if init_ok:
        db_posts = get_all_posts()
        if not db_posts.empty:
            return db_posts

    # Fallback to CSV if database could not be loaded
    sample_path = os.path.join("data", "sample_posts.csv")
    if not os.path.exists(sample_path):
        sample_path = "sample_posts.csv"

    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        df = pd.DataFrame([
            {"post_id": 1, "timestamp": "2026-09-01 10:00:00", "author": "sample_user", "text": "Great platform and awesome support! #Tech", "source": "Twitter"},
            {"post_id": 2, "timestamp": "2026-09-01 11:00:00", "author": "commuter", "text": "Terrible traffic jam today. Bakwas roads! #Traffic", "source": "Twitter"}
        ])
    return analyze_dataframe(df)


if "dataset" not in st.session_state:
    st.session_state.dataset = load_bundled_data()


# Pool of simulated incoming posts
SIMULATION_POSTS_POOL = [
    {"author": "rahul_dev", "source": "Twitter", "text": "The new update completely fixed the memory leak. Extremely pleased with this release! #TechUpdate #Engineering"},
    {"author": "anita_blr", "source": "Twitter", "text": "Traffic on Old Airport road is standstill for 45 minutes. Signal is dead again! #BangaloreTraffic #TrafficJam"},
    {"author": "foodie_karan", "source": "Instagram", "text": "Tried the viral cheese paratha today. Bilkul bekaar laga, waste of money! #FoodFail #Disappointed"},
    {"author": "cricket_fever", "source": "Twitter", "text": "Unbelievable Yorker in the final over! Match winning performance right there. Zabardast game! #IPL #Cricket"},
    {"author": "deepa_travels", "source": "Instagram", "text": "Watching the sunset over Dal Lake in Srinagar. Pure serenity and calm. #Kashmir #TravelGram #Peace"},
    {"author": "arun_fintech", "source": "LinkedIn", "text": "Support ticket was resolved within 5 minutes and payment status updated. Smooth customer service! #FinTech #Support"},
    {"author": "sneha_study", "source": "Twitter", "text": "Exam hall server crashed right during online test! Students are terrified of losing an entire semester. #CollegeExam #Fear"},
    {"author": "vikas_p", "source": "Twitter", "text": "The movie ending was so unexpected and thrilling! Did not see that twist coming. #Cinema #Entertainment"},
    {"author": "neha_r", "source": "Twitter", "text": "Courier boy dumped the expensive electronic goods in the mud outside. Pathetic delivery service! #CustomerServiceFail #Angry"},
    {"author": "amit_c", "source": "LinkedIn", "text": "Parliament session passed the clean energy transition guidelines. Constructive policy step forward. #Policy #CleanEnergy"}
]


# Semantic Theme System
is_dark = (st.session_state.theme == "Dark")

THEME_VARS = {
    "dark": {
        "bg_primary": "#0f172a",
        "bg_secondary": "#111827",
        "bg_card": "#1e293b",
        "bg_card_hover": "#273549",
        "text_primary": "#f8fafc",
        "text_secondary": "#94a3b8",
        "text_muted": "#64748b",
        "border": "#334155",
        "button_bg": "#1e293b",
        "button_text": "#f8fafc",
        "button_border": "#334155",
        "button_hover": "#334155",
        "button_hover_text": "#38bdf8",
        "button_hover_border": "#475569",
        "button_focus_border": "#38bdf8",
        "button_focus_ring": "rgba(56, 189, 248, 0.25)",
        "nav_active_bg": "#0c4a6e",
        "nav_active_border": "#38bdf8",
        "nav_active_text": "#f0f9ff",
        "accent_blue": "#38bdf8",
        "accent_green": "#10b981",
        "accent_red": "#ef4444",
        "plot_grid": "#334155",
        "plot_font": "#94a3b8"
    },
    "light": {
        "bg_primary": "#f8fafc",
        "bg_secondary": "#ffffff",
        "bg_card": "#ffffff",
        "bg_card_hover": "#f1f5f9",
        "border": "#cbd5e1",
        "text_primary": "#0f172a",
        "text_secondary": "#475569",
        "text_muted": "#64748b",
        "button_bg": "#ffffff",
        "button_text": "#0f172a",
        "button_border": "#cbd5e1",
        "button_hover": "#f1f5f9",
        "button_hover_text": "#0284c7",
        "button_hover_border": "#94a3b8",
        "button_focus_border": "#0284c7",
        "button_focus_ring": "rgba(2, 132, 199, 0.20)",
        "nav_active_bg": "#e0f2fe",
        "nav_active_border": "#0284c7",
        "nav_active_text": "#0369a1",
        "accent_blue": "#0284c7",
        "accent_green": "#059669",
        "accent_red": "#dc2626",
        "plot_grid": "#e2e8f0",
        "plot_font": "#475569"
    }
}
t = THEME_VARS["dark"] if is_dark else THEME_VARS["light"]

sidebar_css_display = "block" if st.session_state.sidebar_visible else "none"
sidebar_css_width = "300px" if st.session_state.sidebar_visible else "0px"

st.markdown(f"""
<style>
    /* Semantic CSS Variables */
    :root {{
        --bg-primary: {t['bg_primary']};
        --bg-secondary: {t['bg_secondary']};
        --bg-card: {t['bg_card']};
        --bg-card-hover: {t['bg_card_hover']};
        --border-color: {t['border']};
        --text-primary: {t['text_primary']};
        --text-secondary: {t['text_secondary']};
        --text-muted: {t['text_muted']};
        --button-bg: {t['button_bg']};
        --button-text: {t['button_text']};
        --button-border: {t['button_border']};
        --button-hover: {t['button_hover']};
        --button-hover-text: {t['button_hover_text']};
        --button-hover-border: {t['button_hover_border']};
        --button-focus-border: {t['button_focus_border']};
        --button-focus-ring: {t['button_focus_ring']};
        --nav-active-bg: {t['nav_active_bg']};
        --nav-active-border: {t['nav_active_border']};
        --nav-active-text: {t['nav_active_text']};
        --accent-blue: {t['accent_blue']};
        --accent-green: {t['accent_green']};
        --accent-red: {t['accent_red']};
    }}

    /* Global Body and App Reset */
    html, body, .stApp {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        overflow-x: hidden !important;
    }}

    /* Streamlit Main Header and Toolbar */
    header[data-testid="stHeader"] {{
        background-color: var(--bg-primary) !important;
    }}

    /* Centered Dashboard Container */
    .main .block-container {{
        max-width: 1440px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
        transition: max-width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    /* Collapsible Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: {sidebar_css_display} !important;
        min-width: {sidebar_css_width} !important;
        max-width: 320px !important;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        background-color: var(--bg-secondary) !important;
    }}

    /* Hide default branding footer */
    footer {{visibility: hidden; display: none !important;}}
    #MainMenu {{visibility: visible;}}

    /* Universal Theme-Aware Button Rules (Covers normal, hover, active, focus states) */
    button,
    .stButton > button,
    .stDownloadButton > button,
    button[kind="secondary"],
    button[kind="primary"],
    button[kind="header"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button,
    .stFileUploader button,
    div[data-testid="stButton"] button {{
        background-color: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--button-border) !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.45rem 0.9rem !important;
        box-shadow: none !important;
        white-space: nowrap !important;
        transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
    }}

    /* Universal Text Color inside all buttons */
    button *,
    .stButton > button *,
    .stDownloadButton > button *,
    button div,
    button div p,
    button div span,
    button p,
    button span {{
        color: inherit !important;
    }}

    /* Hover State for all buttons */
    button:hover,
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {{
        background-color: var(--button-hover) !important;
        color: var(--button-hover-text) !important;
        border-color: var(--button-hover-border) !important;
    }}

    /* Focus & Active State: Prevents solid black buttons in Light Mode */
    button:focus,
    button:focus-visible,
    button:active,
    .stButton > button:focus,
    .stButton > button:focus-visible,
    .stButton > button:active,
    button[kind="secondary"]:focus,
    button[kind="secondary"]:focus-visible,
    button[kind="secondary"]:active,
    button[data-testid="stBaseButton-secondary"]:focus,
    button[data-testid="stBaseButton-secondary"]:focus-visible,
    button[data-testid="stBaseButton-secondary"]:active {{
        background-color: var(--button-hover) !important;
        color: var(--button-hover-text) !important;
        border-color: var(--button-focus-border) !important;
        outline: none !important;
        box-shadow: 0 0 0 2px var(--button-focus-ring) !important;
    }}

    /* Header Bar Layout */
    .header-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        padding-bottom: 1rem;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid var(--border-color);
    }}
    .header-title-box {{
        display: flex;
        flex-direction: column;
    }}
    .header-title {{
        font-size: clamp(1.35rem, 2.2vw, 1.85rem);
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.25;
    }}
    .header-subtitle {{
        font-size: clamp(0.82rem, 1.1vw, 0.92rem);
        color: var(--text-secondary);
        margin-top: 4px;
        margin-bottom: 0;
    }}

    /* Responsive KPI Grid: 5 cards Desktop, 2-3 Tablet, 1 Mobile */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
        width: 100%;
        margin-bottom: 1.5rem;
        box-sizing: border-box;
    }}
    @media (max-width: 1120px) {{
        .kpi-grid {{
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }}
    }}
    @media (max-width: 768px) {{
        .kpi-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }}
    }}
    @media (max-width: 500px) {{
        .kpi-grid {{
            grid-template-columns: 1fr;
        }}
    }}

    /* KPI Card styling with uniform height and text protection */
    .kpi-card {{
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        min-height: 110px;
        box-sizing: border-box;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }}
    .kpi-card:hover {{
        border-color: var(--accent-blue);
    }}
    .kpi-label {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .kpi-value {{
        font-size: clamp(1.4rem, 2vw, 1.85rem);
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
        margin: 6px 0 2px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        color: var(--text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* Card Wrapper for Charts and Sections */
    .chart-card {{
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 1.5rem;
        box-sizing: border-box;
        height: 100%;
        overflow: hidden;
    }}
    .chart-card-title {{
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 12px;
        letter-spacing: -0.01em;
    }}

    /* Sidebar Section Divider and Headers */
    .sidebar-section {{
        padding: 12px 0 6px 0;
        border-top: 1px solid var(--border-color);
        margin-top: 12px;
    }}
    .sidebar-section-title {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin-bottom: 10px;
    }}

    /* Responsive Segmented Navigation Bar */
    .nav-wrapper {{
        margin-bottom: 1.5rem;
        width: 100%;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        background-color: var(--bg-card) !important;
        padding: 6px;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px;
        width: 100%;
        box-sizing: border-box;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
        flex: 1 1 auto;
        min-width: 140px;
        text-align: center;
        justify-content: center;
        padding: 8px 14px !important;
        margin: 0 !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
        transition: all 0.15s ease !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label * {{
        color: var(--text-secondary) !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
        background-color: var(--bg-card-hover) !important;
        color: var(--text-primary) !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover * {{
        color: var(--text-primary) !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
        background-color: var(--nav-active-bg) !important;
        border: 1px solid var(--nav-active-border) !important;
        color: var(--nav-active-text) !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] *,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {{
        color: var(--nav-active-text) !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] input[type="radio"] {{
        display: none !important;
    }}

    /* Simulation Status Banner */
    .simulation-banner {{
        background-color: rgba(2, 132, 199, 0.10);
        border-left: 4px solid var(--accent-blue);
        color: var(--text-primary);
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }}

    /* Anomaly Spike Card */
    .spike-banner {{
        background-color: rgba(239, 68, 68, 0.10);
        border: 1px solid var(--accent-red);
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 1.25rem;
        color: var(--text-primary);
        font-size: 0.88rem;
    }}

    /* Legal & Governance Container */
    .legal-box {{
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 24px;
        color: var(--text-primary);
        font-size: 0.90rem;
        line-height: 1.65;
    }}

    /* Form inputs and text areas */
    input[type="text"],
    textarea,
    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="input"],
    [data-baseweb="textarea"],
    [data-baseweb="base-input"] {{
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
    }}

    /* Multiselect and tags */
    [data-baseweb="tag"] {{
        background-color: var(--bg-card-hover) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    [data-baseweb="tag"] * {{
        color: var(--text-primary) !important;
    }}

    /* File Uploader Dropzone */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: var(--bg-card) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: 6px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: var(--text-secondary) !important;
    }}

    /* Dataframe and Tables */
    [data-testid="stDataFrame"] {{
        width: 100% !important;
        border-radius: 6px !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--bg-card) !important;
    }}

    /* Captions and subtext */
    .stCaption,
    [data-testid="stCaptionContainer"] {{
        color: var(--text-secondary) !important;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px !important;
    }}
    .streamlit-expanderContent {{
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
    }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR / CONTROL CENTER
# ==========================================
with st.sidebar:
    # Sidebar Header with Controls
    sb_h_col1, sb_h_col2 = st.columns([3, 1])
    with sb_h_col1:
        st.markdown(f"<h3 style='margin:0; font-size:1.15rem; color: var(--text-primary);'>Control Center</h3>", unsafe_allow_html=True)
    with sb_h_col2:
        if st.button("✕", help="Minimize Control Center"):
            st.session_state.sidebar_visible = False
            st.rerun()

    st.caption("Configure dataset ingestion, live stream simulation, and display preferences.")

    # Section 1: Quick Actions (Theme, Reset, Settings)
    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Preferences</div></div>", unsafe_allow_html=True)
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        theme_btn_label = "Switch to Light" if is_dark else "Switch to Dark"
        if st.button(theme_btn_label, use_container_width=True, help="Toggle application theme"):
            st.session_state.theme = "Light" if is_dark else "Dark"
            st.rerun()
    with act_col2:
        if st.button("Reset All", use_container_width=True, help="Reset dataset and dashboard state"):
            st.session_state.dataset = load_bundled_data()
            st.session_state.sim_active = False
            st.session_state.sim_last_post = None
            st.success("Dashboard reset to default.")
            st.rerun()

    # Section 2: Data Source
    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Data Source</div></div>", unsafe_allow_html=True)
    db_stats = get_db_stats()
    st.caption(f"SQLite DB: **{db_stats['total_posts']} posts** ({db_stats['status']})")

    data_source_mode = st.radio(
        "Select Source",
        options=["SQLite Database (socialpulse.db)", "Upload Custom CSV"],
        index=0,
        label_visibility="collapsed"
    )

    if data_source_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV (Max 5MB)",
            type=["csv"],
            help="Upload a UTF-8 CSV containing a text column."
        )
        if uploaded_file is not None:
            is_valid, err_msg, parsed_df = validate_uploaded_file(uploaded_file)
            if not is_valid:
                st.error(err_msg)
            else:
                if st.button("Ingest & Store in Database", use_container_width=True):
                    with st.spinner("Analyzing CSV & storing in SQLite database..."):
                        analyzed = analyze_dataframe(parsed_df)
                        insert_posts_batch(analyzed)
                        st.session_state.dataset = get_all_posts()
                        st.success(f"Stored {len(analyzed)} posts in SQLite database.")
                        st.rerun()

    if st.button("Reload from SQLite Database", use_container_width=True):
        st.session_state.dataset = get_all_posts()
        st.session_state.sim_last_post = None
        st.success("Synchronized from SQLite database.")
        st.rerun()

    # Section 3: Live Simulation
    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Live Simulation</div></div>", unsafe_allow_html=True)
    sim_toggle = st.toggle("Enable Live Stream", value=st.session_state.sim_active)
    if sim_toggle != st.session_state.sim_active:
        st.session_state.sim_active = sim_toggle
        st.rerun()

    sim_speed = st.slider("Stream Interval (seconds)", min_value=2, max_value=10, value=3)

    # Section 4: Documentation & Compliance Shortcut
    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Governance</div></div>", unsafe_allow_html=True)
    if st.button("View Policies & Terms", use_container_width=True):
        st.session_state.current_view = "Compliance & Policies"
        st.rerun()

    # Optional Settings Expander
    with st.expander("System Settings", expanded=False):
        st.write(f"**Theme:** {st.session_state.theme}")
        st.write(f"**Database:** {db_stats['path']}")
        st.write(f"**Database Size:** {db_stats['size_kb']} KB")
        st.write(f"**Database Health:** {db_stats['status']}")
        st.write(f"**Stored Posts:** {db_stats['total_posts']}")
        st.write(f"**Active Session Posts:** {len(st.session_state.dataset)}")
        st.write(f"**Rate Limit Remaining:** {st.session_state.rate_limiter.max_requests - len(st.session_state.rate_limiter.timestamps)}")
        if st.button("Reset & Re-seed Database", use_container_width=True, help="Re-initializes SQLite database with default posts"):
            ok, msg = reset_db()
            if ok:
                st.session_state.dataset = get_all_posts()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# ==========================================
# TOP HEADER & CONTROL TOGGLE
# ==========================================
h_col1, h_col2 = st.columns([3.5, 2])
with h_col1:
    st.markdown("""
    <div class="header-title-box">
        <h1 class="header-title">SocialPulse: AI Sentiment & Topic Intelligence</h1>
        <p class="header-subtitle">Understand what people are saying, how they feel, and what is trending.</p>
    </div>
    """, unsafe_allow_html=True)

with h_col2:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        toggle_label = "☰ Controls" if not st.session_state.sidebar_visible else "✕ Controls"
        if st.button(toggle_label, use_container_width=True, help="Toggle Control Center"):
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible
            st.rerun()
    with btn_col2:
        top_theme_label = "Theme: Dark" if is_dark else "Theme: Light"
        if st.button(top_theme_label, use_container_width=True, help="Toggle Dark/Light Mode"):
            st.session_state.theme = "Light" if is_dark else "Dark"
            st.rerun()


# ==========================================
# LIVE SIMULATION WORKER (if active)
# ==========================================
if st.session_state.sim_active:
    st.markdown(
        '<div class="simulation-banner"><strong>LIVE SIMULATION ACTIVE:</strong> '
        'Ingesting synthetic social posts into memory to demonstrate real-time KPI, emotion, and trend calculations. '
        'This is a synthetic demonstration, not an external API stream.</div>',
        unsafe_allow_html=True
    )
    
    # Process simulated post
    allowed, limit_msg = st.session_state.rate_limiter.allow_request()
    if allowed:
        new_sample = random.choice(SIMULATION_POSTS_POOL)
        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        analyzed_row = analyze_single_post(
            text=new_sample["text"],
            timestamp=now_ts,
            author=new_sample["author"],
            post_id=len(st.session_state.dataset) + 1,
            source=new_sample["source"]
        )
        insert_post(analyzed_row)
        new_row_df = pd.DataFrame([analyzed_row])
        new_row_df["parsed_timestamp"] = pd.to_datetime(new_row_df["timestamp"], errors="coerce")
        st.session_state.dataset = pd.concat([st.session_state.dataset, new_row_df], ignore_index=True)
        st.session_state.sim_last_post = analyzed_row

    if st.session_state.sim_last_post:
        lp = st.session_state.sim_last_post
        st.info(
            f"Latest Ingested [{lp['timestamp']}] @{lp['author']} on {lp['source']}: "
            f'"{lp["raw_text"]}" | Sentiment: **{lp["sentiment"]}** | Emotion: **{lp["emotion"]}** | Topic: **{lp["topic"]}**'
        )


# ==========================================
# COMPUTE DYNAMIC METRICS
# ==========================================
df_current = st.session_state.dataset
sent_summary = get_sentiment_summary(df_current)
emo_summary = get_emotion_summary(df_current)
top_summary = get_topic_summary(df_current)
kw_summary = get_keyword_trends(df_current, top_n=10)
hashtag_summary = get_hashtag_trends(df_current, top_n=10)
time_summary = get_sentiment_over_time(df_current, freq="D")
spikes = detect_sentiment_spikes(df_current)
top_topic_name = top_summary.iloc[0]["Topic"] if not top_summary.empty else "N/A"


# ==========================================
# RESPONSIVE TOP NAVIGATION
# ==========================================
nav_options = [
    "Analytics Overview",
    "Post Explorer",
    "Single Post Analyzer",
    "Compliance & Policies"
]

selected_view = st.radio(
    "Navigation Menu",
    options=nav_options,
    index=nav_options.index(st.session_state.current_view) if st.session_state.current_view in nav_options else 0,
    horizontal=True,
    label_visibility="collapsed"
)
if selected_view != st.session_state.current_view:
    st.session_state.current_view = selected_view
    st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ==========================================
# VIEW 1: ANALYTICS OVERVIEW
# ==========================================
if st.session_state.current_view == "Analytics Overview":

    # 1. Responsive KPI Grid
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total Posts</div>
            <div class="kpi-value">{sent_summary['total_posts']:,}</div>
            <div class="kpi-sub">Analyzed corpus</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Positive</div>
            <div class="kpi-value" style="color: var(--accent-green);">{sent_summary['positive_pct']}%</div>
            <div class="kpi-sub">{sent_summary['positive_count']} posts</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Negative</div>
            <div class="kpi-value" style="color: var(--accent-red);">{sent_summary['negative_pct']}%</div>
            <div class="kpi-sub">{sent_summary['negative_count']} posts</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Neutral</div>
            <div class="kpi-value" style="color: var(--text-secondary);">{sent_summary['neutral_pct']}%</div>
            <div class="kpi-sub">{sent_summary['neutral_count']} posts</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Top Topic</div>
            <div class="kpi-value" style="font-size: clamp(1.15rem, 1.5vw, 1.45rem); color: var(--accent-blue);">{top_topic_name}</div>
            <div class="kpi-sub">Leading category</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Anomaly Alerts
    if spikes:
        for spike in spikes:
            st.markdown(f"""
            <div class="spike-banner">
                <strong>Sentiment Spike Alert [{spike['date']}]:</strong> {spike['message']}
            </div>
            """, unsafe_allow_html=True)

    # 3. Charts Row 1: Sentiment Donut & Emotion Breakdown
    row1_col1, row1_col2 = st.columns([1, 1.25], gap="medium")

    with row1_col1:
        st.markdown("<div class='chart-card'><div class='chart-card-title'>Sentiment Distribution</div>", unsafe_allow_html=True)
        sent_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [sent_summary["positive_count"], sent_summary["negative_count"], sent_summary["neutral_count"]]
        })
        fig_donut = px.pie(
            sent_df,
            names="Sentiment",
            values="Count",
            hole=0.55,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#10b981",
                "Negative": "#ef4444",
                "Neutral": "#64748b"
            }
        )
        fig_donut.update_layout(
            height=320,
            margin=dict(l=15, r=15, t=15, b=25),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["plot_font"])
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown("<div class='chart-card'><div class='chart-card-title'>Emotion Breakdown</div>", unsafe_allow_html=True)
        fig_emo = px.bar(
            emo_summary,
            x="Emotion",
            y="Count",
            color="Emotion",
            color_discrete_map={
                "Happiness": "#10b981",
                "Anger": "#dc2626",
                "Sadness": "#3b82f6",
                "Frustration": "#f59e0b",
                "Fear": "#0284c7",
                "Surprise": "#06b6d4",
                "Neutral": "#64748b"
            },
            text="Count"
        )
        fig_emo.update_layout(
            height=320,
            margin=dict(l=15, r=15, t=15, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["plot_font"]),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=t["plot_grid"]),
            showlegend=False
        )
        st.plotly_chart(fig_emo, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Charts Row 2: Sentiment Trends Over Time
    st.markdown("<div class='chart-card'><div class='chart-card-title'>Sentiment Trends Over Time</div>", unsafe_allow_html=True)
    if not time_summary.empty and len(time_summary) > 1:
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=time_summary["Time"], y=time_summary["Positive"],
            mode="lines+markers", name="Positive",
            line=dict(color="#10b981", width=2.5)
        ))
        fig_time.add_trace(go.Scatter(
            x=time_summary["Time"], y=time_summary["Negative"],
            mode="lines+markers", name="Negative",
            line=dict(color="#ef4444", width=2.5)
        ))
        fig_time.add_trace(go.Scatter(
            x=time_summary["Time"], y=time_summary["Neutral"],
            mode="lines+markers", name="Neutral",
            line=dict(color="#64748b", width=2, dash="dot")
        ))
        fig_time.update_layout(
            height=320,
            margin=dict(l=15, r=15, t=15, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["plot_font"]),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=t["plot_grid"], title="Post Count"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Multiple timestamp intervals needed to graph timeline trends.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. Charts Row 3: Topics & Salient Keywords
    row3_col1, row3_col2 = st.columns(2, gap="medium")

    with row3_col1:
        st.markdown("<div class='chart-card'><div class='chart-card-title'>Topic Volume</div>", unsafe_allow_html=True)
        fig_topic = px.bar(
            top_summary.head(8),
            x="Count",
            y="Topic",
            orientation="h",
            color_discrete_sequence=["#2563eb"],
            text="Count"
        )
        fig_topic.update_layout(
            height=300,
            yaxis=dict(autorange="reversed", showgrid=False),
            xaxis=dict(showgrid=True, gridcolor=t["plot_grid"]),
            margin=dict(l=15, r=15, t=15, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["plot_font"])
        )
        st.plotly_chart(fig_topic, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row3_col2:
        st.markdown("<div class='chart-card'><div class='chart-card-title'>Trending Keywords & Hashtags</div>", unsafe_allow_html=True)
        sub_tab1, sub_tab2 = st.tabs(["Keywords (TF-IDF)", "Hashtags"])
        with sub_tab1:
            if not kw_summary.empty:
                fig_kw = px.bar(
                    kw_summary.head(8),
                    x="Score",
                    y="Keyword",
                    orientation="h",
                    color_discrete_sequence=["#0284c7"],
                    text="Score"
                )
                fig_kw.update_layout(
                    height=240,
                    yaxis=dict(autorange="reversed", showgrid=False),
                    xaxis=dict(showgrid=True, gridcolor=t["plot_grid"]),
                    margin=dict(l=10, r=10, t=10, b=15),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=t["plot_font"])
                )
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No keywords detected.")

        with sub_tab2:
            if not hashtag_summary.empty:
                fig_ht = px.bar(
                    hashtag_summary.head(8),
                    x="Count",
                    y="Hashtag",
                    orientation="h",
                    color_discrete_sequence=["#0d9488"],
                    text="Count"
                )
                fig_ht.update_layout(
                    height=240,
                    yaxis=dict(autorange="reversed", showgrid=False),
                    xaxis=dict(showgrid=True, gridcolor=t["plot_grid"]),
                    margin=dict(l=10, r=10, t=10, b=15),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=t["plot_font"])
                )
                st.plotly_chart(fig_ht, use_container_width=True)
            else:
                st.info("No hashtags detected in current dataset.")
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# VIEW 2: POST EXPLORER
# ==========================================
elif st.session_state.current_view == "Post Explorer":
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-card-title'>Social Post Explorer & Query Engine</div>", unsafe_allow_html=True)
    st.caption("Search, filter, and inspect processed social posts with classification confidence.")

    # Responsive Filter Row
    f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 1.5])
    with f_col1:
        sent_filter = st.multiselect("Sentiment", options=["Positive", "Negative", "Neutral"], default=[])
    with f_col2:
        all_emos = sorted(df_current["emotion"].unique().tolist())
        emo_filter = st.multiselect("Emotion", options=all_emos, default=[])
    with f_col3:
        all_tops = sorted(df_current["topic"].unique().tolist())
        top_filter = st.multiselect("Topic", options=all_tops, default=[])
    with f_col4:
        search_query = st.text_input("Text Search", value="", placeholder="Search text or username...")

    filtered_df = df_current.copy()
    if sent_filter:
        filtered_df = filtered_df[filtered_df["sentiment"].isin(sent_filter)]
    if emo_filter:
        filtered_df = filtered_df[filtered_df["emotion"].isin(emo_filter)]
    if top_filter:
        filtered_df = filtered_df[filtered_df["topic"].isin(top_filter)]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["raw_text"].astype(str).str.lower().str.contains(q, na=False) |
            filtered_df["author"].astype(str).str.lower().str.contains(q, na=False)
        ]

    st.markdown(f"<p style='margin: 10px 0; color: var(--text-secondary); font-size: 0.88rem;'>Showing <strong>{len(filtered_df)}</strong> of {len(df_current)} posts</p>", unsafe_allow_html=True)

    display_df = filtered_df[[
        "timestamp", "raw_text", "sentiment", "emotion", "topic", "sentiment_confidence"
    ]].rename(columns={
        "timestamp": "Time",
        "raw_text": "Post",
        "sentiment": "Sentiment",
        "emotion": "Emotion",
        "topic": "Topic",
        "sentiment_confidence": "Confidence"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.2f",
                min_value=0.0,
                max_value=1.0
            ),
            "Post": st.column_config.TextColumn("Post", width="large")
        }
    )

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Results as CSV",
        data=csv_bytes,
        file_name="socialpulse_filtered_posts.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# VIEW 3: SINGLE POST ANALYZER
# ==========================================
elif st.session_state.current_view == "Single Post Analyzer":
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-card-title'>Interactive Single Post Live Analyzer</div>", unsafe_allow_html=True)
    st.caption("Test the multilingual NLP pipeline with English, Hinglish, emojis, hashtags, or negation statements.")

    input_text = st.text_area(
        "Enter Post Content",
        height=110,
        placeholder="Type here... e.g., Silk board traffic was completely jammed today! Bilkul bekaar experience. #BangaloreTraffic",
        max_chars=500
    )

    analyze_btn = st.button("Run Analysis", help="Execute complete NLP pipeline on input post")

    if analyze_btn and input_text:
        sanitized = sanitize_input_text(input_text)
        result = analyze_single_post(sanitized)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: var(--text-primary); margin-bottom: 12px;'>Analysis Results</h4>", unsafe_allow_html=True)

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Sentiment", result["sentiment"], delta=f"Conf: {result['sentiment_confidence']}")
        with res_col2:
            st.metric("Emotion", result["emotion"], delta=f"Conf: {result['emotion_confidence']}")
        with res_col3:
            st.metric("Topic", result["topic"], delta=f"Conf: {result['topic_confidence']}")
        with res_col4:
            st.metric("Sentiment Score", result["sentiment_score"])

        st.markdown(f"**Cleaned Tokens:** `{result['cleaned_text']}`")
        if result["hashtags"]:
            st.markdown(f"**Extracted Hashtags:** `{' '.join([f'#{h}' for h in result['hashtags']])}`")
        if result["keywords"]:
            st.markdown(f"**Salient Keywords:** `{' '.join(result['keywords'])}`")
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# VIEW 4: COMPLIANCE & POLICIES
# ==========================================
elif st.session_state.current_view == "Compliance & Policies":
    st.markdown("""
    <div class="legal-box">
        <h3 style="margin-top: 0; font-size: 1.25rem; color: var(--text-primary);">Platform Governance, Privacy Policy & Terms</h3>
        <p style="color: var(--text-secondary);"><strong>Effective Date:</strong> September 11, 2026</p>
        <hr style="border-color: var(--border-color); margin: 16px 0;">
        <h4 style="font-size: 1.05rem; color: var(--text-primary);">1. Privacy Policy</h4>
        <p style="color: var(--text-secondary);">SocialPulse is designed as an in-memory analytics platform prioritizing data protection:</p>
        <ul style="color: var(--text-secondary);">
            <li><strong>In-Memory Processing:</strong> All user-uploaded CSV files and interactive text tests are held exclusively in temporary memory during the active session. No user posts or extracted metrics are persisted or transmitted to third-party tracking services.</li>
            <li><strong>Input Sanitization:</strong> Strict input validation enforces file size thresholds (5MB maximum) and strips script injection vectors.</li>
            <li><strong>Fictional Demonstration Data:</strong> The bundled dataset and real-time simulation streams consist entirely of synthetic, non-PII test scenarios created for hackathon analysis.</li>
        </ul>
        <hr style="border-color: var(--border-color); margin: 16px 0;">
        <h4 style="font-size: 1.05rem; color: var(--text-primary);">2. Terms and Conditions</h4>
        <ul style="color: var(--text-secondary);">
            <li><strong>Analytical Purpose:</strong> Classifications are generated through transparent lexical rules, negation windows, and frequency heuristics. Outputs are provided for analytics and monitoring demonstration.</li>
            <li><strong>Content Responsibility:</strong> Users uploading external datasets are responsible for ensuring content conforms to relevant copyright and digital governance laws.</li>
            <li><strong>Resource Rate Limiting:</strong> Interactive analysis requests are rate-limited to maintain server responsiveness and stability.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# SIMULATION TIMER LOOP (if enabled)
# ==========================================
if st.session_state.sim_active:
    time.sleep(sim_speed)
    st.rerun()
