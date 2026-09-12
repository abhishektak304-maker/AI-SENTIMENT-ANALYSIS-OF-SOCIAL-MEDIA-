"""SocialPulse: AI Sentiment & Topic Intelligence Dashboard.

A responsive, production-grade analytical dashboard for social media sentiment,
emotion, topic, and trend analysis supporting English and Hinglish.
Features comprehensive theme awareness (Dark & Light), collapsible Control Center,
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
    page_icon="favicon.png" if os.path.exists("favicon.png") else None,
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
    init_ok, _ = init_db()
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


# Semantic Theme System (Slate dark / clean light)
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
sidebar_margin_left = "0px" if st.session_state.sidebar_visible else "-340px"

# Comprehensive, zero-overflow, responsive design system
st.markdown(f"""
<style>
    /* Semantic CSS Design Tokens */
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

    /* Global Body and Viewport Reset: Prevent unwanted horizontal scrolling */
    html, body, .stApp {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        overflow-x: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }}

    *, *:before, *:after {{
        box-sizing: inherit;
    }}

    /* Streamlit Main Header and Toolbar */
    header[data-testid="stHeader"] {{
        background-color: var(--bg-primary) !important;
        border-bottom: 1px solid var(--border-color) !important;
        height: 3rem !important;
        z-index: 50 !important;
    }}

    /* Centered Dashboard Container with responsive padding and fluid max-width */
    .main .block-container {{
        max-width: 1440px !important;
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: clamp(0.75rem, 2vw, 2rem) !important;
        padding-right: clamp(0.75rem, 2vw, 2rem) !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
        width: 100% !important;
        transition: padding 0.2s ease, max-width 0.2s ease;
    }}

    /* Collapsible Sidebar Styling with smooth CSS width transition */
    [data-testid="stSidebar"] {{
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
        transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1), margin-left 0.25s ease, opacity 0.2s ease !important;
        display: {sidebar_css_display} !important;
        width: {sidebar_css_width} !important;
        min-width: 0 !important;
        max-width: 320px !important;
        margin-left: {sidebar_margin_left} !important;
        overflow: hidden !important;
        flex-shrink: 0 !important;
        z-index: 100 !important;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.25rem !important;
        padding-left: 1.15rem !important;
        padding-right: 1.15rem !important;
        padding-bottom: 2rem !important;
        background-color: var(--bg-secondary) !important;
        width: 300px !important;
    }}

    /* Native collapse button styling */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {{
        background-color: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--button-border) !important;
        border-radius: 4px !important;
    }}

    /* Hide default branding footer */
    footer {{visibility: hidden; display: none !important;}}
    #MainMenu {{visibility: visible;}}

    /* Theme-Aware Button Rules: strictly 4px radius, no pill buttons, no purple */
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
    .stFileUploader button,
    div[data-testid="stButton"] button {{
        background-color: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--button-border) !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.42rem 0.85rem !important;
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

    /* Focus & Active State: Prevents solid black buttons */
    button:focus,
    button:focus-visible,
    button:active,
    .stButton > button:focus,
    .stButton > button:focus-visible,
    .stButton > button:active {{
        background-color: var(--button-hover) !important;
        color: var(--button-hover-text) !important;
        border-color: var(--button-focus-border) !important;
        outline: none !important;
        box-shadow: 0 0 0 2px var(--button-focus-ring) !important;
    }}

    /* Top Action Bar Layout */
    .top-action-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        width: 100%;
        box-sizing: border-box;
    }}

    .header-title-box {{
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .header-title {{
        font-size: clamp(1.25rem, 2vw, 1.75rem);
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.25;
    }}

    .header-subtitle {{
        font-size: clamp(0.80rem, 1vw, 0.90rem);
        color: var(--text-secondary);
        margin-top: 3px;
        margin-bottom: 0;
        line-height: 1.4;
    }}

    /* Guarantee header buttons never collapse or overflow */
    .header-btn-group {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
    }}

    .header-btn-group div[data-testid="stHorizontalBlock"] {{
        gap: 8px !important;
        flex-wrap: nowrap !important;
    }}

    .header-btn-group div[data-testid="column"] {{
        min-width: 95px !important;
        flex: 1 1 auto !important;
    }}

    .header-btn-group button {{
        min-width: 90px !important;
        width: 100% !important;
        text-align: center !important;
    }}

    /* Responsive KPI Grid: 5 cards Desktop, 3-2 Tablet, 1 Mobile */
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
            gap: 12px;
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
            gap: 8px;
        }}
    }}

    /* KPI Card styling with uniform height and text protection */
    .kpi-card {{
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 112px;
        height: 100%;
        box-sizing: border-box;
        overflow: hidden;
        transition: border-color 0.15s ease;
    }}
    .kpi-card:hover {{
        border-color: var(--accent-blue);
    }}
    .kpi-label {{
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .kpi-value {{
        font-size: clamp(1.35rem, 1.8vw, 1.85rem);
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
        margin-bottom: 1.25rem;
        box-sizing: border-box;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }}
    .chart-card-title {{
        font-size: 0.98rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 12px;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    /* Sidebar Section Divider and Headers */
    .sidebar-section {{
        padding: 12px 0 6px 0;
        border-top: 1px solid var(--border-color);
        margin-top: 12px;
    }}
    .sidebar-section-title {{
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Responsive Segmented Navigation Bar */
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
        flex: 1 1 160px;
        min-width: 140px;
        text-align: center;
        justify-content: center;
        padding: 8px 12px !important;
        margin: 0 !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        font-size: 0.86rem !important;
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
        margin-bottom: 1.25rem;
        font-size: 0.88rem;
        line-height: 1.5;
        width: 100%;
        box-sizing: border-box;
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
        width: 100%;
        box-sizing: border-box;
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
        width: 100%;
        box-sizing: border-box;
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

    /* Mobile Drawer breakpoint */
    @media (max-width: 768px) {{
        [data-testid="stSidebar"] {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            z-index: 9999 !important;
            box-shadow: 0 0 25px rgba(0,0,0,0.5) !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR / CONTROL CENTER
# ==========================================
with st.sidebar:
    # Header: Title & Collapse Button
    sb_h_col1, sb_h_col2 = st.columns([3, 1], vertical_alignment="center")
    with sb_h_col1:
        st.markdown("<h3 style='margin:0; font-size:1.15rem; color: var(--text-primary);'>Control Center</h3>", unsafe_allow_html=True)
    with sb_h_col2:
        if st.button("✕", help="Minimize Control Center", key="sidebar_close_btn"):
            st.session_state.sidebar_visible = False
            st.rerun()

    st.caption("Manage datasets, live stream simulation, and system preferences.")

    # Section 1: Useful Sidebar Controls (Theme, Reset, Settings)
    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Quick Actions</div></div>", unsafe_allow_html=True)
    sb_action_c1, sb_action_c2 = st.columns(2, gap="small")
    with sb_action_c1:
        theme_btn_label = "Switch to Light" if is_dark else "Switch to Dark"
        if st.button(theme_btn_label, use_container_width=True, help="Toggle Theme", key="sb_theme_toggle_btn"):
            st.session_state.theme = "Light" if is_dark else "Dark"
            st.rerun()
    with sb_action_c2:
        if st.button("Reset All", use_container_width=True, help="Reset dataset and dashboard state", key="sb_reset_dashboard_btn"):
            st.session_state.dataset = load_bundled_data()
            st.session_state.sim_active = False
            st.session_state.sim_last_post = None
            st.success("Dashboard reset to default.")
            st.rerun()

    # Section 2: Data Source
    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Data Source</div></div>", unsafe_allow_html=True)
    db_stats = get_db_stats()
    st.caption(f"Active DB: **{db_stats['total_posts']} posts** ({db_stats['status']})")

    data_source_mode = st.radio(
        "Select Source",
        options=["Bundled Dataset (SQLite)", "Upload Custom CSV"],
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
                if st.button("Ingest & Store in Database", use_container_width=True, key="ingest_csv_btn"):
                    with st.spinner("Analyzing CSV & storing in SQLite database..."):
                        analyzed = analyze_dataframe(parsed_df)
                        insert_posts_batch(analyzed)
                        st.session_state.dataset = get_all_posts()
                        st.success(f"Stored {len(analyzed)} posts in SQLite database.")
                        st.rerun()

    if st.button("Reset to Default Sample Dataset", use_container_width=True, key="reload_db_btn"):
        st.session_state.dataset = load_bundled_data()
        st.session_state.sim_last_post = None
        st.success("Synchronized with bundled dataset.")
        st.rerun()

    # Section 3: Live Simulation
    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Live Simulation</div></div>", unsafe_allow_html=True)
    sim_toggle = st.toggle("Enable Live Simulation", value=st.session_state.sim_active, key="live_sim_toggle")
    if sim_toggle != st.session_state.sim_active:
        st.session_state.sim_active = sim_toggle
        st.rerun()

    sim_speed = st.slider("Simulation Interval (seconds)", min_value=2, max_value=10, value=3, key="sim_speed_slider")

    # Section 4: Documentation & Legal Shortcut
    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Documentation & Legal</div></div>", unsafe_allow_html=True)
    if st.button("Privacy Policy & Terms", use_container_width=True, key="sb_legal_nav_btn"):
        st.session_state.current_view = "Compliance & Policies"
        st.rerun()

    # Section 5: Optional Settings Expander
    with st.expander("System Settings", expanded=False):
        st.write(f"**Theme:** {st.session_state.theme}")
        st.write(f"**Database:** {db_stats['path']}")
        st.write(f"**Database Size:** {db_stats['size_kb']} KB")
        st.write(f"**Database Health:** {db_stats['status']}")
        st.write(f"**Stored Posts:** {db_stats['total_posts']}")
        st.write(f"**Active Session Posts:** {len(st.session_state.dataset)}")
        st.write(f"**Rate Limit Remaining:** {st.session_state.rate_limiter.max_requests - len(st.session_state.rate_limiter.timestamps)}")
        if st.button("Re-seed Database", use_container_width=True, help="Re-initializes SQLite database with default posts", key="sb_reseed_db_btn"):
            ok, msg = reset_db()
            if ok:
                st.session_state.dataset = get_all_posts()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# ==========================================
# TOP HEADER & CONTROL TOGGLE BAR
# ==========================================
header_col1, header_col2 = st.columns([3.2, 1.3], vertical_alignment="center")

with header_col1:
    st.markdown("""
    <div class="header-title-box">
        <h1 class="header-title">SocialPulse: AI Sentiment &amp; Topic Intelligence</h1>
        <p class="header-subtitle">Real-time sentiment, emotion, and topic intelligence across social posts.</p>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("<div class='header-btn-group'>", unsafe_allow_html=True)
    h_btn1, h_btn2 = st.columns(2, gap="small")
    with h_btn1:
        toggle_label = "☰ Controls" if not st.session_state.sidebar_visible else "✕ Controls"
        if st.button(toggle_label, use_container_width=True, key="top_sidebar_toggle_btn", help="Collapse or expand the Control Center sidebar"):
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible
            st.rerun()
    with h_btn2:
        top_theme_label = "Dark" if is_dark else "Light"
        if st.button(top_theme_label, use_container_width=True, key="top_theme_toggle_btn", help="Switch Dark / Light theme"):
            st.session_state.theme = "Light" if is_dark else "Dark"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


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

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# ==========================================
# VIEW 1: ANALYTICS OVERVIEW
# ==========================================
if st.session_state.current_view == "Analytics Overview":

    # 1. Responsive KPI Grid: 5 cards Desktop, 3-2 Tablet, 1 Mobile
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

    # 3. Charts Row 1: Sentiment Share & Emotion Breakdown (aligned side-by-side)
    row1_col1, row1_col2 = st.columns([1, 1], gap="medium")

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
            height=330,
            margin=dict(l=15, r=15, t=15, b=25),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["plot_font"]),
            autosize=True
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
            height=330,
            margin=dict(l=15, r=15, t=15, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["plot_font"]),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=t["plot_grid"]),
            showlegend=False,
            autosize=True
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
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            autosize=True
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
            font=dict(color=t["plot_font"]),
            autosize=True
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
                    font=dict(color=t["plot_font"]),
                    autosize=True
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
                    font=dict(color=t["plot_font"]),
                    autosize=True
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
    f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 1.5], gap="small")
    with f_col1:
        sent_filter = st.multiselect("Sentiment", options=["Positive", "Negative", "Neutral"], default=[], key="pe_sent_filter")
    with f_col2:
        all_emos = sorted(df_current["emotion"].unique().tolist())
        emo_filter = st.multiselect("Emotion", options=all_emos, default=[], key="pe_emo_filter")
    with f_col3:
        all_tops = sorted(df_current["topic"].unique().tolist())
        top_filter = st.multiselect("Topic", options=all_tops, default=[], key="pe_top_filter")
    with f_col4:
        search_query = st.text_input("Text Search", value="", placeholder="Search text or author...", key="pe_search_query")

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
        mime="text/csv",
        key="pe_download_csv_btn"
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
        max_chars=500,
        key="single_post_input_area"
    )

    analyze_btn = st.button("Run Analysis", help="Execute complete NLP pipeline on input post", key="run_single_analysis_btn")

    if analyze_btn and input_text:
        sanitized = sanitize_input_text(input_text)
        result = analyze_single_post(sanitized)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: var(--text-primary); margin-bottom: 12px;'>Analysis Results</h4>", unsafe_allow_html=True)

        res_col1, res_col2, res_col3, res_col4 = st.columns(4, gap="small")
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
        <h3 style="margin-top: 0; font-size: 1.25rem; color: var(--text-primary);">Platform Governance, Privacy Policy &amp; Terms</h3>
        <p style="color: var(--text-secondary);"><strong>Effective Date:</strong> September 11, 2026</p>
        <hr style="border-color: var(--border-color); margin: 16px 0;">
        <h4 style="font-size: 1.05rem; color: var(--text-primary);">1. Privacy Policy</h4>
        <p style="color: var(--text-secondary);">SocialPulse is designed as an analytics platform prioritizing data protection:</p>
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
