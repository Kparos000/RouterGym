from __future__ import annotations

import ast
import base64
import html
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "RouterGym" / "results" / "analysis_outputs"
PLOTS_DIR = ANALYSIS_DIR / "plots"
MANUAL_DIR = ANALYSIS_DIR / "manual_audit"
GOLD_DIR = ANALYSIS_DIR / "gold_resolution_eval"
GOLD_PLOTS_DIR = GOLD_DIR / "plots"
SAMPLE_PATH = ANALYSIS_DIR / "streamlit_ticket_sample.csv"
LOGO_PATH = ROOT / "assets" / "emlyon_logo.png"

EMLYON_RED = "#d5001f"
DARK_NAVY = "#172033"
CHARCOAL = "#243041"
MUTED = "#667085"
LIGHT_BG = "#f6f8fb"
SOFT_BLUE = "#eaf2ff"
TEAL = "#138a72"
PURPLE = "#6c4ad2"

CONFIG_ORDER = [
    "llm_only__base_llm1__mem_rag_bm25",
    "llm_only__base_llm2__mem_rag_bm25",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25",
    "slm_only__base_slm1__mem_rag_bm25",
    "slm_only__base_slm2__mem_rag_bm25",
]

CONFIG_LABELS = {
    "llm_only__base_llm1__mem_rag_bm25": "LLM1",
    "llm_only__base_llm2__mem_rag_bm25": "LLM2",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25": "SLM-dom SLM1→LLM2",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25": "SLM-dom SLM2→LLM2",
    "slm_only__base_slm1__mem_rag_bm25": "SLM1",
    "slm_only__base_slm2__mem_rag_bm25": "SLM2",
}

FULL_CONFIG_LABELS = {
    "llm_only__base_llm1__mem_rag_bm25": "LLM-only / LLM1",
    "llm_only__base_llm2__mem_rag_bm25": "LLM-only / LLM2",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25": "SLM-dominant / SLM1→LLM2",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25": "SLM-dominant / SLM2→LLM2",
    "slm_only__base_slm1__mem_rag_bm25": "SLM-only / SLM1",
    "slm_only__base_slm2__mem_rag_bm25": "SLM-only / SLM2",
}

ROUTER_LABELS = {
    "llm_only": "LLM-only",
    "slm_only": "SLM-only",
    "slm_dominant": "SLM-dominant",
}

ROUTER_COLORS = {
    "LLM-only": DARK_NAVY,
    "SLM-only": TEAL,
    "SLM-dominant": EMLYON_RED,
}

PAGES = [
    "Executive Snapshot",
    "Configuration Comparator",
    "Gold and Manual Quality",
    "Cost-Latency-Quality Frontier",
    "Ticket Explorer",
    "Methodology and Reproducibility",
]

SEARCH_DIRS = (
    ANALYSIS_DIR,
    PLOTS_DIR,
    MANUAL_DIR,
    GOLD_DIR,
    GOLD_PLOTS_DIR,
)


st.set_page_config(
    page_title="RouterGym Results Explorer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_global_css_legacy() -> None:
    st.markdown(
        f"""
<style>
    :root {{
        --rg-red: {EMLYON_RED};
        --rg-navy: {DARK_NAVY};
        --rg-charcoal: {CHARCOAL};
        --rg-muted: {MUTED};
        --rg-blue: {SOFT_BLUE};
        --rg-teal: {TEAL};
        --rg-purple: {PURPLE};
        --rg-line: rgba(36, 48, 65, 0.12);
        --rg-card: rgba(255, 255, 255, 0.72);
    }}

    #MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stSidebar"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 8% 8%, rgba(213, 0, 31, 0.08), transparent 26rem),
            radial-gradient(circle at 86% 4%, rgba(70, 130, 230, 0.16), transparent 28rem),
            linear-gradient(135deg, #f9fbff 0%, #eef4fb 42%, #f7f7f5 100%);
        color: var(--rg-charcoal);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .block-container {{
        max-width: 1380px;
        padding: 1.35rem 2rem 3.5rem;
    }}

    h1, h2, h3, h4 {{
        color: var(--rg-navy);
        letter-spacing: 0;
    }}

    p, li, label, div {{
        letter-spacing: 0;
    }}

    .rg-topbar {{
        position: sticky;
        top: 0.75rem;
        z-index: 20;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.8rem;
        padding: 0.78rem 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.74);
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.74);
        box-shadow: 0 18px 55px rgba(23, 32, 51, 0.10);
        backdrop-filter: blur(20px);
    }}

    .rg-brand {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        min-width: 0;
    }}

    .rg-brand-logo {{
        width: 112px;
        height: 42px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(23, 32, 51, 0.08);
        object-fit: contain;
        padding: 0.35rem 0.55rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }}

    .rg-brand-fallback {{
        width: 52px;
        height: 42px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: var(--rg-navy);
        color: white;
        font-weight: 800;
    }}

    .rg-brand-title {{
        display: flex;
        flex-direction: column;
        gap: 0.08rem;
        min-width: 0;
    }}

    .rg-brand-title strong {{
        color: var(--rg-navy);
        font-size: 1.03rem;
        line-height: 1.05;
    }}

    .rg-brand-title span {{
        color: var(--rg-muted);
        font-size: 0.78rem;
        line-height: 1.1;
        white-space: nowrap;
    }}

    .rg-top-badges {{
        display: flex;
        align-items: center;
        justify-content: flex-end;
        flex-wrap: wrap;
        gap: 0.45rem;
    }}

    .rg-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.35rem;
        width: fit-content;
        min-height: 1.85rem;
        padding: 0.32rem 0.68rem;
        border-radius: 999px;
        border: 1px solid rgba(23, 32, 51, 0.10);
        color: var(--rg-charcoal);
        background: rgba(255, 255, 255, 0.78);
        font-size: 0.76rem;
        font-weight: 760;
        line-height: 1;
        white-space: nowrap;
    }}

    .rg-badge-red {{
        color: #9d0018;
        border-color: rgba(213, 0, 31, 0.22);
        background: rgba(213, 0, 31, 0.08);
    }}

    .rg-badge-dark {{
        color: white;
        border-color: rgba(23, 32, 51, 0.2);
        background: var(--rg-navy);
    }}

    .rg-badge-green {{
        color: #0a5f4d;
        border-color: rgba(19, 138, 114, 0.22);
        background: rgba(19, 138, 114, 0.10);
    }}

    .rg-badge-blue {{
        color: #264c83;
        border-color: rgba(45, 105, 185, 0.20);
        background: rgba(70, 130, 230, 0.11);
    }}

    .rg-badge-purple {{
        color: #4e35a5;
        border-color: rgba(108, 74, 210, 0.20);
        background: rgba(108, 74, 210, 0.10);
    }}

    .rg-badge-warning {{
        color: #8c4b00;
        border-color: rgba(196, 112, 20, 0.22);
        background: rgba(250, 176, 65, 0.15);
    }}

    div[role="radiogroup"] {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        width: fit-content;
        max-width: 100%;
        margin: 0.15rem auto 1.3rem auto;
        padding: 0.42rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.70);
        background: rgba(255, 255, 255, 0.58);
        box-shadow: 0 16px 40px rgba(23, 32, 51, 0.08);
        backdrop-filter: blur(18px);
    }}

    div[role="radiogroup"] label {{
        min-height: 2.25rem;
        margin: 0 !important;
        padding: 0.22rem 0.72rem !important;
        border-radius: 999px;
        border: 1px solid transparent;
        color: var(--rg-muted);
        font-size: 0.86rem;
        font-weight: 760;
        transition: all 140ms ease;
    }}

    div[role="radiogroup"] label:hover {{
        color: var(--rg-navy);
        background: rgba(255, 255, 255, 0.72);
        border-color: rgba(23, 32, 51, 0.08);
    }}

    div[role="radiogroup"] label:has(input:checked) {{
        color: white;
        background: linear-gradient(135deg, var(--rg-navy), #31445d);
        box-shadow: 0 12px 28px rgba(23, 32, 51, 0.22);
    }}

    div[role="radiogroup"] label > div:first-child {{
        display: none;
    }}

    .rg-hero {{
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: 1.25fr 0.75fr;
        gap: 2rem;
        min-height: 410px;
        margin: 0.25rem 0 1.6rem;
        padding: 3rem;
        border-radius: 34px;
        border: 1px solid rgba(255,255,255,0.76);
        background:
            linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,255,255,0.50)),
            linear-gradient(145deg, rgba(234,242,255,0.90), rgba(247,247,245,0.84));
        box-shadow: 0 30px 90px rgba(23, 32, 51, 0.14);
        backdrop-filter: blur(18px);
    }}

    .rg-hero:before {{
        content: "";
        position: absolute;
        inset: -34% -18% auto 46%;
        height: 440px;
        background:
            radial-gradient(circle, rgba(213,0,31,0.16), transparent 52%),
            radial-gradient(circle at 32% 70%, rgba(19,138,114,0.16), transparent 48%);
        filter: blur(14px);
        pointer-events: none;
    }}

    .rg-hero-copy {{
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 1rem;
    }}

    .rg-kicker {{
        color: var(--rg-red);
        font-size: 0.78rem;
        font-weight: 860;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    .rg-hero h1 {{
        max-width: 830px;
        margin: 0;
        color: var(--rg-navy);
        font-size: clamp(2.5rem, 6.5vw, 5.15rem);
        line-height: 0.94;
        font-weight: 880;
    }}

    .rg-hero h2 {{
        margin: 0;
        color: #3a4657;
        font-size: clamp(1.18rem, 2vw, 1.62rem);
        font-weight: 760;
    }}

    .rg-hero p {{
        max-width: 770px;
        margin: 0;
        color: #566273;
        font-size: 1.02rem;
        line-height: 1.7;
    }}

    .rg-hero-panel {{
        position: relative;
        z-index: 1;
        align-self: stretch;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 1rem;
        padding: 1.35rem;
        border-radius: 28px;
        border: 1px solid rgba(255,255,255,0.72);
        background: rgba(255,255,255,0.58);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.72), 0 24px 60px rgba(23,32,51,0.10);
    }}

    .rg-hero-logo {{
        width: 148px;
        max-width: 100%;
        border-radius: 18px;
        padding: 0.65rem 0.8rem;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(23,32,51,0.08);
        object-fit: contain;
    }}

    .rg-hero-stat-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
    }}

    .rg-hero-stat {{
        padding: 1rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.66);
        border: 1px solid rgba(23,32,51,0.08);
    }}

    .rg-hero-stat strong {{
        display: block;
        color: var(--rg-navy);
        font-size: 1.45rem;
        line-height: 1;
    }}

    .rg-hero-stat span {{
        display: block;
        margin-top: 0.35rem;
        color: var(--rg-muted);
        font-size: 0.78rem;
        font-weight: 720;
    }}

    .rg-section-header {{
        margin: 2.1rem 0 1rem;
    }}

    .rg-section-header .rg-kicker {{
        margin-bottom: 0.35rem;
    }}

    .rg-section-header h2 {{
        margin: 0;
        font-size: clamp(1.65rem, 3vw, 2.45rem);
        line-height: 1.05;
    }}

    .rg-section-header p {{
        max-width: 850px;
        margin: 0.65rem 0 0;
        color: #5b6779;
        line-height: 1.65;
    }}

    .rg-metric-card,
    .rg-insight-card,
    .rg-filter-shell,
    .rg-table-shell,
    .rg-answer-card,
    .rg-pipeline-card,
    .rg-storage-card {{
        border: 1px solid rgba(255,255,255,0.76);
        background: var(--rg-card);
        box-shadow: 0 18px 48px rgba(23,32,51,0.09);
        backdrop-filter: blur(18px);
    }}

    .rg-metric-card {{
        min-height: 142px;
        margin-bottom: 1rem;
        padding: 1.25rem;
        border-radius: 24px;
    }}

    .rg-metric-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.1rem;
        height: 2.1rem;
        margin-bottom: 0.9rem;
        border-radius: 14px;
        color: var(--rg-red);
        background: rgba(213,0,31,0.08);
        font-size: 0.83rem;
        font-weight: 850;
    }}

    .rg-metric-label {{
        color: var(--rg-muted);
        font-size: 0.78rem;
        font-weight: 780;
        text-transform: uppercase;
    }}

    .rg-metric-value {{
        margin-top: 0.35rem;
        color: var(--rg-navy);
        font-size: clamp(1.55rem, 3vw, 2.18rem);
        font-weight: 880;
        line-height: 1.05;
    }}

    .rg-metric-caption {{
        margin-top: 0.55rem;
        color: #667085;
        font-size: 0.86rem;
        line-height: 1.45;
    }}

    .rg-insight-card {{
        min-height: 170px;
        margin-bottom: 1rem;
        padding: 1.35rem;
        border-radius: 24px;
        border-left: 5px solid rgba(70,130,230,0.55);
    }}

    .rg-insight-card h3 {{
        margin: 0 0 0.55rem;
        font-size: 1.05rem;
    }}

    .rg-insight-card p {{
        margin: 0;
        color: #586476;
        line-height: 1.62;
        font-size: 0.94rem;
    }}

    .rg-insight-success {{
        border-left-color: var(--rg-teal);
    }}

    .rg-insight-warning {{
        border-left-color: #d18a18;
    }}

    .rg-insight-danger {{
        border-left-color: var(--rg-red);
    }}

    .rg-insight-purple {{
        border-left-color: var(--rg-purple);
    }}

    .rg-filter-shell,
    .rg-table-shell {{
        margin: 0.4rem 0 1.2rem;
        padding: 1.1rem;
        border-radius: 24px;
    }}

    .rg-filter-title {{
        margin: 0 0 0.9rem;
        color: var(--rg-navy);
        font-size: 0.96rem;
        font-weight: 820;
    }}

    .rg-chart-title {{
        margin: 0.2rem 0 0.55rem;
        padding: 0 0.15rem;
    }}

    .rg-chart-title h3 {{
        margin: 0;
        font-size: 1.03rem;
    }}

    .rg-chart-title p {{
        margin: 0.35rem 0 0;
        color: var(--rg-muted);
        font-size: 0.84rem;
        line-height: 1.45;
    }}

    div[data-testid="stPlotlyChart"] {{
        margin-bottom: 1.05rem;
        padding: 1rem;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.76);
        background: rgba(255,255,255,0.74);
        box-shadow: 0 18px 48px rgba(23,32,51,0.09);
        backdrop-filter: blur(18px);
    }}

    div[data-testid="stImage"] img {{
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.76);
        box-shadow: 0 18px 48px rgba(23,32,51,0.09);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.38rem;
        background: rgba(255,255,255,0.56);
        border: 1px solid rgba(255,255,255,0.72);
    }}

    .stTabs [data-baseweb="tab"] {{
        min-height: 2.35rem;
        border-radius: 999px;
        color: var(--rg-muted);
        font-weight: 760;
    }}

    .stTabs [aria-selected="true"] {{
        color: var(--rg-navy);
        background: rgba(255,255,255,0.86);
        box-shadow: 0 8px 20px rgba(23,32,51,0.08);
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid rgba(23,32,51,0.10);
        border-radius: 18px;
        background: rgba(255,255,255,0.54);
        overflow: hidden;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(23,32,51,0.10);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 12px 34px rgba(23,32,51,0.06);
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 24px;
        border-color: rgba(255,255,255,0.76);
        background: rgba(255,255,255,0.58);
        box-shadow: 0 18px 48px rgba(23,32,51,0.08);
        backdrop-filter: blur(18px);
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {{
        border-radius: 14px !important;
        border-color: rgba(23,32,51,0.12) !important;
        background: rgba(255,255,255,0.78) !important;
    }}

    .rg-answer-card {{
        margin: 0 0 1rem;
        padding: 1.15rem;
        border-radius: 24px;
    }}

    .rg-answer-head {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.85rem;
    }}

    .rg-answer-head h3 {{
        margin: 0;
        font-size: 1.05rem;
    }}

    .rg-answer-subtitle {{
        margin-top: 0.25rem;
        color: var(--rg-muted);
        font-size: 0.82rem;
        line-height: 1.35;
    }}

    .rg-answer-badges {{
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.38rem;
    }}

    .rg-pipeline-grid,
    .rg-storage-grid,
    .rg-check-grid {{
        display: grid;
        gap: 0.85rem;
    }}

    .rg-pipeline-grid {{
        grid-template-columns: repeat(7, minmax(0, 1fr));
    }}

    .rg-storage-grid,
    .rg-check-grid {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }}

    .rg-pipeline-card,
    .rg-storage-card {{
        min-height: 112px;
        padding: 1rem;
        border-radius: 22px;
    }}

    .rg-pipeline-card span,
    .rg-storage-card span,
    .rg-check-item span {{
        color: var(--rg-muted);
        font-size: 0.74rem;
        font-weight: 820;
        text-transform: uppercase;
    }}

    .rg-pipeline-card strong,
    .rg-storage-card strong,
    .rg-check-item strong {{
        display: block;
        margin-top: 0.5rem;
        color: var(--rg-navy);
        line-height: 1.25;
    }}

    .rg-check-item {{
        padding: 0.95rem;
        border-radius: 20px;
        border: 1px solid rgba(19,138,114,0.18);
        background: rgba(255,255,255,0.62);
    }}

    @media (max-width: 980px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .rg-topbar,
        .rg-answer-head {{
            align-items: flex-start;
            flex-direction: column;
        }}

        .rg-hero {{
            grid-template-columns: 1fr;
            min-height: unset;
            padding: 2rem;
        }}

        .rg-hero-stat-grid,
        .rg-storage-grid,
        .rg-check-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        .rg-pipeline-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        div[role="radiogroup"] {{
            width: 100%;
            border-radius: 24px;
        }}
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def is_dark_mode() -> bool:
    return bool(st.session_state.get("dark_mode", False))


def theme_values(dark_mode: bool | None = None) -> dict[str, str]:
    dark = is_dark_mode() if dark_mode is None else dark_mode
    if dark:
        return {
            "bg": "#08111f",
            "bg_soft": "#111c2e",
            "panel": "rgba(18, 29, 47, 0.78)",
            "panel_strong": "rgba(24, 38, 59, 0.94)",
            "text": "#f5f7fb",
            "muted": "#aab7ca",
            "border": "rgba(203, 213, 225, 0.18)",
            "shadow": "0 24px 64px rgba(0, 0, 0, 0.34)",
            "accent": EMLYON_RED,
            "accent_soft": "rgba(213, 0, 31, 0.20)",
            "danger": "#ff6b7f",
            "success": "#4ed8b8",
            "warning": "#f6c76b",
            "nav_active": EMLYON_RED,
            "nav_inactive": "#e8edf7",
            "plot_bg": "rgba(18, 29, 47, 0.74)",
            "llm": "#8fb3ff",
            "slm": "#4ed8b8",
            "slm_dom": "#ff667a",
        }
    return {
        "bg": "#f9fbff",
        "bg_soft": "#eef4fb",
        "panel": "rgba(255, 255, 255, 0.72)",
        "panel_strong": "rgba(255, 255, 255, 0.88)",
        "text": DARK_NAVY,
        "muted": MUTED,
        "border": "rgba(36, 48, 65, 0.12)",
        "shadow": "0 18px 48px rgba(23, 32, 51, 0.09)",
        "accent": EMLYON_RED,
        "accent_soft": "rgba(213, 0, 31, 0.08)",
        "danger": EMLYON_RED,
        "success": TEAL,
        "warning": "#d18a18",
        "nav_active": DARK_NAVY,
        "nav_inactive": CHARCOAL,
        "plot_bg": "rgba(255, 255, 255, 0.74)",
        "llm": DARK_NAVY,
        "slm": TEAL,
        "slm_dom": EMLYON_RED,
    }


def inject_global_css(dark_mode: bool | None = None) -> None:
    theme = theme_values(dark_mode)
    st.markdown(
        f"""
<style>
    :root {{
        --bg: {theme["bg"]};
        --bg-soft: {theme["bg_soft"]};
        --panel: {theme["panel"]};
        --panel-strong: {theme["panel_strong"]};
        --text: {theme["text"]};
        --muted: {theme["muted"]};
        --border: {theme["border"]};
        --shadow: {theme["shadow"]};
        --accent: {theme["accent"]};
        --accent-soft: {theme["accent_soft"]};
        --danger: {theme["danger"]};
        --success: {theme["success"]};
        --warning: {theme["warning"]};
        --nav-active: {theme["nav_active"]};
        --nav-inactive: {theme["nav_inactive"]};
        --plot-bg: {theme["plot_bg"]};
    }}

    #MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stSidebar"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 8% 8%, rgba(213, 0, 31, 0.10), transparent 26rem),
            radial-gradient(circle at 86% 4%, rgba(70, 130, 230, 0.16), transparent 28rem),
            linear-gradient(135deg, var(--bg) 0%, var(--bg-soft) 44%, var(--bg) 100%) !important;
        color: var(--text) !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .block-container {{
        max-width: 1380px;
        padding: 1.35rem 2rem 3.5rem;
    }}

    h1, h2, h3, h4,
    .rg-brand-title strong,
    .rg-hero h1,
    .rg-hero h2,
    .rg-section-header h2,
    .rg-metric-value,
    .rg-chart-title h3,
    .rg-answer-head h3,
    .rg-pipeline-card strong,
    .rg-storage-card strong,
    .rg-check-item strong {{
        color: var(--text) !important;
        letter-spacing: 0;
    }}

    p, li, label, span, div {{
        letter-spacing: 0;
    }}

    .rg-topbar,
    .rg-hero,
    .rg-hero-panel,
    .rg-metric-card,
    .rg-insight-card,
    .rg-filter-shell,
    .rg-table-shell,
    .rg-answer-card,
    .rg-pipeline-card,
    .rg-storage-card,
    .rg-check-item,
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid var(--border) !important;
        background: var(--panel) !important;
        box-shadow: var(--shadow) !important;
        backdrop-filter: blur(18px);
    }}

    .rg-topbar {{
        position: sticky;
        top: 0.75rem;
        z-index: 20;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.8rem;
        padding: 0.78rem 0.9rem;
        border-radius: 24px;
    }}

    .rg-brand,
    .rg-top-badges,
    .rg-answer-head {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }}

    .rg-brand-logo,
    .rg-hero-logo {{
        background: rgba(255, 255, 255, 0.86) !important;
        border: 1px solid var(--border) !important;
        object-fit: contain;
    }}

    .rg-brand-logo {{
        width: 112px;
        height: 42px;
        border-radius: 14px;
        padding: 0.35rem 0.55rem;
    }}

    .rg-brand-title {{
        display: flex;
        flex-direction: column;
        gap: 0.08rem;
        min-width: 0;
    }}

    .rg-brand-title span,
    .rg-hero p,
    .rg-hero-stat span,
    .rg-section-header p,
    .rg-metric-label,
    .rg-metric-caption,
    .rg-insight-card p,
    .rg-chart-title p,
    .rg-answer-subtitle,
    .rg-pipeline-card span,
    .rg-storage-card span,
    .rg-check-item span {{
        color: var(--muted) !important;
    }}

    .rg-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: fit-content;
        min-height: 1.85rem;
        padding: 0.32rem 0.68rem;
        border-radius: 999px;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        background: var(--panel-strong) !important;
        font-size: 0.76rem;
        font-weight: 760;
        line-height: 1;
        white-space: nowrap;
    }}

    .rg-badge-dark {{
        color: #ffffff !important;
        background: var(--nav-active) !important;
    }}

    .rg-badge-red {{
        color: var(--danger) !important;
        background: var(--accent-soft) !important;
    }}

    .rg-badge-green {{
        color: var(--success) !important;
    }}

    .rg-badge-warning {{
        color: var(--warning) !important;
    }}

    div[role="radiogroup"] {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        width: fit-content;
        max-width: 100%;
        margin: 0.15rem auto 1.3rem auto;
        padding: 0.42rem;
        border-radius: 999px;
        border: 1px solid var(--border) !important;
        background: var(--panel) !important;
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
    }}

    div[role="radiogroup"] label {{
        min-height: 2.25rem;
        margin: 0 !important;
        padding: 0.22rem 0.72rem !important;
        border-radius: 999px;
        border: 1px solid transparent;
        color: var(--nav-inactive) !important;
        font-size: 0.86rem;
        font-weight: 760;
        transition: all 140ms ease;
    }}

    div[role="radiogroup"] label *,
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span {{
        color: var(--nav-inactive) !important;
        -webkit-text-fill-color: var(--nav-inactive) !important;
    }}

    div[role="radiogroup"] label:hover {{
        color: var(--text) !important;
        background: var(--panel-strong) !important;
        border-color: var(--border) !important;
    }}

    div[role="radiogroup"] label:hover *,
    div[role="radiogroup"] label:hover p,
    div[role="radiogroup"] label:hover span {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }}

    div[role="radiogroup"] label:has(input:checked) {{
        color: #ffffff !important;
        background: linear-gradient(135deg, var(--nav-active), #31445d) !important;
        box-shadow: 0 12px 28px rgba(23, 32, 51, 0.28);
    }}

    div[role="radiogroup"] label:has(input:checked) *,
    div[role="radiogroup"] label:has(input:checked) p,
    div[role="radiogroup"] label:has(input:checked) span {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}

    div[role="radiogroup"] label > div:first-child {{
        display: none;
    }}

    .rg-hero {{
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: 1.25fr 0.75fr;
        gap: 2rem;
        min-height: 410px;
        margin: 0.25rem 0 1.6rem;
        padding: 3rem;
        border-radius: 34px;
    }}

    .rg-kicker,
    .rg-metric-icon {{
        color: var(--accent) !important;
    }}

    .rg-metric-icon {{
        background: var(--accent-soft) !important;
    }}

    .rg-insight-card {{
        border-left: 5px solid rgba(70,130,230,0.60) !important;
    }}

    .rg-insight-success {{
        border-left-color: var(--success) !important;
    }}

    .rg-insight-warning {{
        border-left-color: var(--warning) !important;
    }}

    .rg-insight-danger {{
        border-left-color: var(--danger) !important;
    }}

    .rg-insight-purple {{
        border-left-color: #9d8cff !important;
    }}

    div[data-testid="stPlotlyChart"] {{
        margin-bottom: 1.05rem;
        padding: 1rem;
        border-radius: 24px;
        border: 1px solid var(--border) !important;
        background: var(--plot-bg) !important;
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.38rem;
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        min-height: 2.35rem;
        border-radius: 999px;
        color: var(--muted) !important;
        font-weight: 760;
    }}

    .stTabs [aria-selected="true"] {{
        color: var(--text) !important;
        background: var(--panel-strong) !important;
        box-shadow: 0 8px 20px rgba(23,32,51,0.12);
    }}

    div[data-testid="stExpander"],
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border) !important;
        background: var(--panel) !important;
        color: var(--text) !important;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {{
        border-radius: 14px !important;
        border-color: var(--border) !important;
        background: var(--panel-strong) !important;
        color: var(--text) !important;
    }}

    [data-baseweb="tag"] {{
        background: var(--accent-soft) !important;
        color: var(--text) !important;
        border: 1px solid rgba(213, 0, 31, 0.24) !important;
        border-radius: 999px !important;
    }}

    [data-baseweb="tag"] span {{
        color: var(--text) !important;
    }}

    .rg-flow-track {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.7rem 0 1.2rem;
    }}

    .rg-flow-stage {{
        padding: 0.85rem;
        min-height: 82px;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: var(--panel);
    }}

    .rg-flow-stage-active {{
        border-color: rgba(213, 0, 31, 0.42);
        background: var(--accent-soft) !important;
    }}

    .rg-flow-stage span {{
        display: block;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 820;
        text-transform: uppercase;
    }}

    .rg-flow-stage strong {{
        display: block;
        margin-top: 0.35rem;
        color: var(--text);
        line-height: 1.2;
    }}

    @media (max-width: 980px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .rg-topbar,
        .rg-answer-head {{
            align-items: flex-start;
            flex-direction: column;
        }}

        .rg-hero,
        .rg-flow-track {{
            grid-template-columns: 1fr;
        }}

        .rg-hero {{
            min-height: unset;
            padding: 2rem;
        }}

        div[role="radiogroup"] {{
            width: 100%;
            border-radius: 24px;
        }}
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def stretch_width() -> dict[str, object]:
    try:
        major, minor, *_ = (int(part) for part in st.__version__.split(".")[:2])
    except ValueError:
        return {"use_container_width": True}
    if (major, minor) >= (1, 50):
        return {"width": "stretch"}
    return {"use_container_width": True}


def image_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/png;base64,{encoded}"


def render_badge(text: str, variant: str = "neutral") -> str:
    return f'<span class="rg-badge rg-badge-{escape(variant)}">{escape(text)}</span>'


def render_top_nav(current_page: str) -> str:
    selected_page = st.session_state.get("rg_page", current_page)
    if selected_page not in PAGES:
        selected_page = current_page
        st.session_state["rg_page"] = current_page
    st.session_state.setdefault("dark_mode", False)

    logo_uri = image_data_uri(LOGO_PATH)
    logo_html = (
        f'<img class="rg-brand-logo" src="{logo_uri}" alt="emlyon business school logo">'
        if logo_uri
        else '<div class="rg-brand-fallback">RG</div>'
    )
    st.markdown(
        f"""
<div class="rg-topbar">
    <div class="rg-brand">
        {logo_html}
        <div class="rg-brand-title">
            <strong>RouterGym Results Explorer</strong>
            <span>emlyon dissertation benchmark demo</span>
        </div>
    </div>
    <div class="rg-top-badges">
        {render_badge("Offline / Static Results", "red")}
        {render_badge("No API Keys", "blue")}
        {render_badge("No Live Inference", "green")}
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    _, settings_col = st.columns([0.82, 0.18])
    with settings_col:
        st.toggle("Dark mode", key="dark_mode")

    return st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(selected_page),
        horizontal=True,
        key="rg_page",
        label_visibility="collapsed",
    )


def render_hero() -> None:
    logo_uri = image_data_uri(LOGO_PATH)
    logo_html = (
        f'<img class="rg-hero-logo" src="{logo_uri}" alt="emlyon business school logo">'
        if logo_uri
        else '<div class="rg-brand-fallback">RG</div>'
    )
    st.markdown(
        f"""
<section class="rg-hero">
    <div class="rg-hero-copy">
        <div class="rg-kicker">Modern academic AI benchmark</div>
        <h1>RouterGym Results Explorer</h1>
        <h2>From LLM-First to SLM-Dominant</h2>
        <p>
            A static, reproducible product-quality explorer for the final RouterGym benchmark:
            operational cost, latency, parse reliability, deterministic gold scoring, and
            blinded manual audit evidence across six BM25-RAG configurations.
        </p>
        <div class="rg-top-badges">
            {render_badge("Offline / Static Results", "red")}
            {render_badge("No API Keys", "blue")}
            {render_badge("No Live Inference", "green")}
            {render_badge("60k Outputs", "dark")}
        </div>
    </div>
    <aside class="rg-hero-panel">
        <div>{logo_html}</div>
        <div class="rg-hero-stat-grid">
            <div class="rg-hero-stat"><strong>60k</strong><span>generated outputs</span></div>
            <div class="rg-hero-stat"><strong>6</strong><span>final configurations</span></div>
            <div class="rg-hero-stat"><strong>456</strong><span>gold-scored outputs</span></div>
            <div class="rg-hero-stat"><strong>BM25</strong><span>fixed RAG memory</span></div>
        </div>
    </aside>
</section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: str, icon: str | None = None) -> None:
    icon_html = f'<div class="rg-metric-icon">{escape(icon)}</div>' if icon else ""
    st.markdown(
        f"""
<article class="rg-metric-card">
    {icon_html}
    <div class="rg-metric-label">{escape(label)}</div>
    <div class="rg-metric-value">{escape(value)}</div>
    <div class="rg-metric-caption">{escape(caption)}</div>
</article>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(
    items: Sequence[tuple[str, str, str] | tuple[str, str, str, str]],
    columns: int = 3,
) -> None:
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, items[start : start + columns]):
            label, value, caption, *icon = item
            with col:
                render_metric_card(label, value, caption, icon[0] if icon else None)


def render_insight_card(title: str, body: str, variant: str = "info") -> None:
    st.markdown(
        f"""
<article class="rg-insight-card rg-insight-{escape(variant)}">
    <h3>{escape(title)}</h3>
    <p>{escape(body)}</p>
</article>
        """,
        unsafe_allow_html=True,
    )


def render_chart_card(title: str, fig, caption: str | None = None) -> None:
    st.markdown(
        f"""
<div class="rg-chart-title">
    <h3>{escape(title)}</h3>
    {f"<p>{escape(caption)}</p>" if caption else ""}
</div>
        """,
        unsafe_allow_html=True,
    )
    if fig is None:
        render_insight_card(
            "Chart unavailable", "The required result file or metric column is missing.", "warning"
        )
        return
    st.plotly_chart(
        fig,
        config={"displayModeBar": False, "responsive": True},
        **stretch_width(),
    )


def render_section_header(kicker: str, title: str, body: str | None = None) -> None:
    body_html = f"<p>{escape(body)}</p>" if body else ""
    st.markdown(
        f"""
<section class="rg-section-header">
    <div class="rg-kicker">{escape(kicker)}</div>
    <h2>{escape(title)}</h2>
    {body_html}
</section>
        """,
        unsafe_allow_html=True,
    )


def find_file(possible_paths: Sequence[str | Path]) -> Path | None:
    for raw_path in possible_paths:
        path = Path(raw_path)
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(ROOT / path)
            candidates.extend(search_dir / path for search_dir in SEARCH_DIRS)
            candidates.extend(search_dir / path.name for search_dir in SEARCH_DIRS)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        pattern = str(raw_path).replace("\\", "/")
        if any(token in pattern for token in "*?[]"):
            for search_dir in SEARCH_DIRS:
                matches = sorted(search_dir.rglob(pattern))
                if matches:
                    return matches[0]
        else:
            for search_dir in SEARCH_DIRS:
                matches = sorted(search_dir.rglob(path.name))
                if matches:
                    return matches[0]
    return None


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


def safe_read_csv(path: str | Path, *, warn: bool = False) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        if warn:
            render_insight_card(
                "Missing file", f"`{csv_path.relative_to(ROOT)}` is not available.", "warning"
            )
        return pd.DataFrame()
    try:
        return _read_csv_cached(str(csv_path)).copy()
    except Exception as exc:
        if warn:
            render_insight_card(
                "Unreadable file", f"`{csv_path.name}` could not be read: {exc}", "warning"
            )
        return pd.DataFrame()


def config_color_map() -> dict[str, str]:
    theme = theme_values()
    return {
        "LLM-only": theme["llm"],
        "SLM-only": theme["slm"],
        "SLM-dominant": theme["slm_dom"],
    }


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def router_family_from_config(config_identifier: object) -> str:
    text = str(config_identifier or "")
    if text.startswith("llm_only"):
        return "llm_only"
    if text.startswith("slm_only"):
        return "slm_only"
    if text.startswith("slm_dominant"):
        return "slm_dominant"
    return ""


def config_label(config_identifier: object, *, full: bool = False) -> str:
    text = str(config_identifier or "")
    labels = FULL_CONFIG_LABELS if full else CONFIG_LABELS
    return labels.get(text, text.replace("__", " / ").replace("_", " "))


def parse_config_parts(config_identifier: object) -> dict[str, str]:
    text = str(config_identifier or "")
    base = re.search(r"__base_([^_]+)", text)
    esc = re.search(r"__esc_([^_]+)", text)
    mem = re.search(r"__mem_(.+)$", text)
    return {
        "base_model_key": base.group(1) if base else "",
        "escalation_model_key": esc.group(1) if esc else "none",
        "memory_mode": mem.group(1) if mem else "",
    }


def standardize_config_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    config_col = first_existing_column(out, ["analysis_config", "config_identifier", "config"])
    out["analysis_config"] = out[config_col].astype(str) if config_col else ""
    out["config_label"] = out["analysis_config"].map(config_label)
    out["config_label_full"] = out["analysis_config"].map(
        lambda value: config_label(value, full=True)
    )

    if "router_family" not in out.columns:
        router_col = first_existing_column(out, ["router_mode", "router"])
        out["router_family"] = out[router_col].astype(str) if router_col else ""
    out["router_family"] = out["router_family"].replace({"": np.nan, "nan": np.nan})
    out["router_family"] = out["router_family"].fillna(
        out["analysis_config"].map(router_family_from_config)
    )
    out["router_family_label"] = (
        out["router_family"].map(ROUTER_LABELS).fillna(out["router_family"])
    )

    for field in ["base_model_key", "escalation_model_key", "memory_mode"]:
        if field not in out.columns:
            out[field] = out["analysis_config"].map(
                lambda value, key=field: parse_config_parts(value)[key]
            )

    rank = {config: index for index, config in enumerate(CONFIG_ORDER)}
    out["_config_rank"] = out["analysis_config"].map(rank).fillna(999)
    return out.sort_values(["_config_rank", "config_label"]).reset_index(drop=True)


def format_currency(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"${number:,.6f}" if abs(number) < 0.01 else f"${number:,.2f}"


def format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.2%}"


def format_ms(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:,.0f} ms"


def format_number(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def has_text(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in {"", "none", "nan", "null"}


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def bool_label(value: object) -> str:
    return "Yes" if truthy(value) else "No"


def parse_list_value(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, dict):
            return [json.dumps(parsed, ensure_ascii=False)]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text]


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def chart_layout(height: int) -> dict:
    theme = theme_values()
    dark = is_dark_mode()
    return {
        "height": height,
        "template": "plotly_dark" if dark else "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": theme["plot_bg"],
        "font": {"family": "Inter, Arial, sans-serif", "color": theme["text"], "size": 13},
        "margin": {"l": 12, "r": 26, "t": 18, "b": 36},
        "legend_title_text": "",
        "hoverlabel": {
            "bgcolor": theme["panel_strong"],
            "font_color": theme["text"],
            "font_size": 12,
        },
    }


def make_horizontal_bar(
    df: pd.DataFrame,
    value_col: str,
    *,
    x_title: str,
    tickformat: str | None = None,
    height: int = 390,
) -> px.bar | None:
    if df.empty or value_col not in df.columns:
        return None

    chart_df = df.copy()
    chart_df[value_col] = numeric_series(chart_df, value_col)
    chart_df = chart_df.dropna(subset=[value_col, "config_label_full"])
    if chart_df.empty:
        return None

    chart_df = chart_df.sort_values("_config_rank", ascending=False)
    fig = px.bar(
        chart_df,
        x=value_col,
        y="config_label_full",
        color="router_family_label",
        color_discrete_map=config_color_map(),
        orientation="h",
        hover_data={
            "analysis_config": True,
            "router_family_label": True,
            value_col: ":.6f" if "cost" in value_col else True,
            "config_label_full": False,
        },
    )
    fig.update_traces(marker_line_color="rgba(255,255,255,0.95)", marker_line_width=1.2)
    fig.update_layout(**chart_layout(height), bargap=0.28)
    fig.update_xaxes(
        title=x_title,
        tickformat=tickformat,
        showgrid=True,
        gridcolor=theme_values()["border"],
    )
    fig.update_yaxes(title="", automargin=True)
    return fig


def make_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    x_title: str,
    y_title: str,
    size: str | None = None,
    x_tickformat: str | None = None,
    y_tickformat: str | None = None,
    height: int = 520,
) -> px.scatter | None:
    if df.empty or x not in df.columns or y not in df.columns:
        return None

    chart_df = df.copy()
    chart_df[x] = numeric_series(chart_df, x)
    chart_df[y] = numeric_series(chart_df, y)
    if size and size in chart_df.columns:
        chart_df[size] = numeric_series(chart_df, size)
    chart_df = chart_df.dropna(subset=[x, y])
    if chart_df.empty:
        return None

    hover_cols = [
        column
        for column in [
            "config_label_full",
            "analysis_config",
            "avg_total_cost_usd",
            "avg_cost_usd",
            "mean_latency_ms",
            "avg_latency_ms",
            "analysis_usable_rate",
            "parse_error_rate",
            "validation_error_rate",
            "mean_overall_manual_quality",
            "mean_overall_gold_quality_score",
            "generated_category_accuracy",
        ]
        if column in chart_df.columns
    ]
    fig = px.scatter(
        chart_df,
        x=x,
        y=y,
        size=size if size in chart_df.columns else None,
        color="router_family_label",
        color_discrete_map=config_color_map(),
        text="config_label",
        hover_data=hover_cols,
        size_max=34,
    )
    fig.update_traces(
        textposition="top center",
        marker={"line": {"width": 1.4, "color": "white"}, "sizemin": 11},
        cliponaxis=False,
    )
    fig.update_layout(**chart_layout(height))
    fig.update_xaxes(
        title=x_title,
        tickformat=x_tickformat,
        showgrid=True,
        gridcolor=theme_values()["border"],
    )
    fig.update_yaxes(
        title=y_title,
        tickformat=y_tickformat,
        showgrid=True,
        gridcolor=theme_values()["border"],
    )
    return fig


def render_static_plot_card(title: str, possible_paths: Sequence[str | Path], caption: str) -> None:
    st.markdown(
        f"""
<div class="rg-chart-title">
    <h3>{escape(title)}</h3>
    <p>{escape(caption)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    path = find_file(possible_paths)
    if path is None:
        render_insight_card(
            "Plot unavailable",
            "The plot image was not found in committed analysis outputs.",
            "warning",
        )
        return
    try:
        st.image(Image.open(path), **stretch_width())
    except Exception as exc:
        render_insight_card(
            "Plot unavailable", f"The plot image could not be rendered: {exc}", "warning"
        )


@st.cache_data(show_spinner=False)
def load_config_comparator() -> pd.DataFrame:
    summary = safe_read_csv(ANALYSIS_DIR / "summary_by_config.csv")
    if summary.empty:
        return pd.DataFrame()

    out = summary.copy()
    projected = safe_read_csv(ANALYSIS_DIR / "projected_47k_cost_by_config.csv")
    if not projected.empty:
        keep = [
            column
            for column in ["analysis_config", "projected_47k_cost_usd", "avg_cost_per_ticket_usd"]
            if column in projected.columns
        ]
        out = out.merge(projected[keep], on="analysis_config", how="left")

    routing = safe_read_csv(ANALYSIS_DIR / "routing_escalation_summary.csv")
    if not routing.empty:
        keep = [
            column for column in ["analysis_config", "escalation_rate"] if column in routing.columns
        ]
        out = out.merge(routing[keep], on="analysis_config", how="left")

    return standardize_config_labels(out)


@st.cache_data(show_spinner=False)
def load_quality_frontier() -> pd.DataFrame:
    summary = load_config_comparator()
    if summary.empty:
        return pd.DataFrame()

    keep = [
        "analysis_config",
        "config_label",
        "config_label_full",
        "router_family",
        "router_family_label",
        "base_model_key",
        "avg_total_cost_usd",
        "mean_latency_ms",
        "analysis_usable_rate",
        "parse_error_rate",
        "validation_error_rate",
        "_config_rank",
    ]
    frontier = summary[[column for column in keep if column in summary.columns]].copy()

    manual = standardize_config_labels(safe_read_csv(MANUAL_DIR / "manual_quality_by_config.csv"))
    if not manual.empty:
        manual_keep = [
            "analysis_config",
            "mean_overall_manual_quality",
            "manual_quality_pass_rate_7",
        ]
        frontier = frontier.merge(
            manual[[column for column in manual_keep if column in manual.columns]],
            on="analysis_config",
            how="left",
        )

    gold = standardize_config_labels(
        safe_read_csv(GOLD_DIR / "gold_resolution_quality_by_config.csv")
    )
    if not gold.empty:
        gold_keep = [
            "analysis_config",
            "mean_overall_gold_quality_score",
            "generated_category_accuracy",
            "pass_rate_070",
            "pass_rate_080",
        ]
        frontier = frontier.merge(
            gold[[column for column in gold_keep if column in gold.columns]],
            on="analysis_config",
            how="left",
        )

    return frontier.sort_values(["_config_rank", "config_label"]).reset_index(drop=True)


def filtered_configs(
    df: pd.DataFrame,
    selected_routers: Sequence[str],
    selected_config_labels: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    if selected_routers:
        out = out[out["router_family_label"].isin(selected_routers)]
    if selected_config_labels:
        out = out[out["config_label"].isin(selected_config_labels)]
    return out.reset_index(drop=True)


def family_filter_row(
    df: pd.DataFrame, *, key_prefix: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    with st.container(border=True):
        st.markdown("**Filter benchmark configurations**")
        col1, col2 = st.columns([1, 1.8])
        router_options = sorted(df["router_family_label"].dropna().unique().tolist())
        config_options = df.sort_values("_config_rank")["config_label"].drop_duplicates().tolist()
        with col1:
            selected_routers = st.multiselect(
                "Router family",
                router_options,
                default=router_options,
                key=f"{key_prefix}_routers",
            )
        with col2:
            selected_configs = st.multiselect(
                "Configuration",
                config_options,
                default=config_options,
                key=f"{key_prefix}_configs",
            )
    return (
        filtered_configs(df, selected_routers, selected_configs),
        selected_routers,
        selected_configs,
    )


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Config": df["config_label_full"],
            "Router family": df["router_family_label"],
            "Avg cost/ticket": df.get("avg_total_cost_usd", pd.Series(index=df.index)).map(
                format_currency
            ),
            "Projected 47k cost": df.get("projected_47k_cost_usd", pd.Series(index=df.index)).map(
                format_currency
            ),
            "Mean latency": df.get("mean_latency_ms", pd.Series(index=df.index)).map(format_ms),
            "Median latency": df.get("median_latency_ms", pd.Series(index=df.index)).map(format_ms),
            "P95 latency": df.get("p95_latency_ms", pd.Series(index=df.index)).map(format_ms),
            "Usable rate": df.get("analysis_usable_rate", pd.Series(index=df.index)).map(
                format_percent
            ),
            "Parse error rate": df.get("parse_error_rate", pd.Series(index=df.index)).map(
                format_percent
            ),
            "Validation error rate": df.get("validation_error_rate", pd.Series(index=df.index)).map(
                format_percent
            ),
            "Escalation rate": df.get("escalation_rate", pd.Series(index=df.index)).map(
                format_percent
            ),
        }
    )


def metric_value(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row:
        return None
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value) or not math.isfinite(float(value)):
        return None
    return float(value)


def display_config(row: pd.Series | None, *, short: bool = True) -> str:
    if row is None:
        return "n/a"
    if short and has_text(row.get("config_label")):
        return str(row.get("config_label"))
    if has_text(row.get("config_label_full")):
        return str(row.get("config_label_full"))
    if has_text(row.get("analysis_config")):
        return config_label(row.get("analysis_config"), full=not short)
    return "n/a"


def best_row(df: pd.DataFrame, column: str, *, highest: bool) -> pd.Series | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    valid = df[values.notna()].copy()
    if valid.empty:
        return None
    valid[column] = pd.to_numeric(valid[column], errors="coerce")
    return valid.sort_values(column, ascending=not highest).iloc[0]


def normalize_for_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series([0.0] * len(values), index=values.index)
    filled = values.fillna(values.median())
    span = filled.max() - filled.min()
    if not math.isfinite(float(span)) or span == 0:
        return pd.Series([0.0] * len(filled), index=filled.index)
    return (filled - filled.min()) / span


def add_operational_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_operational_score"] = (
        normalize_for_score(out.get("analysis_usable_rate", pd.Series(index=out.index)))
        - normalize_for_score(out.get("parse_error_rate", pd.Series(index=out.index)))
        - normalize_for_score(out.get("validation_error_rate", pd.Series(index=out.index)))
        - normalize_for_score(out.get("avg_total_cost_usd", pd.Series(index=out.index)))
        - normalize_for_score(out.get("mean_latency_ms", pd.Series(index=out.index)))
    )
    return out


def summarize_selected_configs(
    df: pd.DataFrame,
    selected_configs: Sequence[str] | None = None,
    quality_signal: str | None = None,
) -> dict[str, pd.Series | None]:
    subset = df.copy()
    if selected_configs:
        label_set = set(selected_configs)
        subset = subset[
            subset["config_label"].isin(label_set) | subset["analysis_config"].isin(label_set)
        ]
    quality_col = None
    if quality_signal == "Gold quality":
        quality_col = "mean_overall_gold_quality_score"
    elif quality_signal == "Manual quality":
        quality_col = "mean_overall_manual_quality"
    scored = add_operational_score(subset) if not subset.empty else subset
    return {
        "cheapest": best_row(subset, "avg_total_cost_usd", highest=False),
        "most_expensive": best_row(subset, "avg_total_cost_usd", highest=True),
        "fastest": best_row(subset, "mean_latency_ms", highest=False),
        "slowest": best_row(subset, "mean_latency_ms", highest=True),
        "highest_usable": best_row(subset, "analysis_usable_rate", highest=True),
        "highest_parse_error": best_row(subset, "parse_error_rate", highest=True),
        "highest_validation_error": best_row(subset, "validation_error_rate", highest=True),
        "best_operational": best_row(scored, "_operational_score", highest=True),
        "highest_quality": best_row(subset, quality_col, highest=True) if quality_col else None,
    }


def classify_tradeoff(row: pd.Series) -> str:
    usable = metric_value(row, "analysis_usable_rate")
    parse = metric_value(row, "parse_error_rate")
    validation = metric_value(row, "validation_error_rate")
    cost = metric_value(row, "avg_total_cost_usd")
    latency = metric_value(row, "mean_latency_ms")
    quality = metric_value(row, "mean_overall_manual_quality") or metric_value(
        row, "mean_overall_gold_quality_score"
    )
    escalation = metric_value(row, "escalation_rate")

    if parse is not None and parse >= 0.03:
        if cost is not None and cost < 0.001:
            return "low cost / reliability risk"
        return "quality-efficient but structurally risky"
    if validation is not None and validation >= 0.01:
        return "quality-efficient but structurally risky"
    if usable is not None and usable >= 0.99 and cost is not None and cost < 0.002:
        return "strong operational trade-off"
    if quality is not None and quality >= 5 and cost is not None and cost >= 0.004:
        return "high quality / high cost"
    if latency is not None and latency < 1800 and quality is not None and quality < 4.5:
        return "fast but lower quality"
    if escalation is not None and escalation > 0.10 and usable is not None and usable >= 0.99:
        return "escalation-heavy but stable"
    return "balanced but context-dependent"


def delta_text(left: float | None, right: float | None, unit: str) -> str:
    if left is None or right is None:
        return "n/a"
    delta = right - left
    pct = ""
    if left:
        pct = f" ({delta / left:+.1%})"
    if unit == "currency":
        return f"{format_currency(delta)}{pct}"
    if unit == "ms":
        return f"{delta:+,.0f} ms{pct}"
    return f"{delta:+.{3 if unit == 'score' else 2}f}{pct}"


def generate_configuration_interpretation(
    df: pd.DataFrame,
    selected_configs: Sequence[str],
    selected_families: Sequence[str],
) -> list[tuple[str, str, str]]:
    summary = summarize_selected_configs(df, selected_configs)
    cheapest = summary["cheapest"]
    fastest = summary["fastest"]
    best = summary["best_operational"]
    parse_risk = summary["highest_parse_error"]
    usable = summary["highest_usable"]

    family_text = ", ".join(selected_families) if selected_families else "the selected families"
    cards = [
        (
            "Selected comparison summary",
            f"{len(df)} selected configuration(s) across {family_text}. The interpretation below is computed from the active filters and does not change benchmark values.",
            "info",
        )
    ]

    if cheapest is not None:
        cards.append(
            (
                "Best selected cost option",
                f"{display_config(cheapest)} is cheapest at {format_currency(cheapest.get('avg_total_cost_usd'))} per ticket and is classified as {classify_tradeoff(cheapest)}.",
                "success",
            )
        )
    if fastest is not None:
        cards.append(
            (
                "Fastest selected option",
                f"{display_config(fastest)} is fastest at {format_ms(fastest.get('mean_latency_ms'))}. Speed is a deployment advantage, not a correctness guarantee.",
                "info",
            )
        )
    if parse_risk is not None:
        risk_variant = (
            "warning" if (metric_value(parse_risk, "parse_error_rate") or 0.0) > 0.01 else "success"
        )
        cards.append(
            (
                "Reliability warning",
                f"{display_config(parse_risk)} has the highest selected parse-error rate at {format_percent(parse_risk.get('parse_error_rate'))}; highest usable rate is {display_config(usable)} at {format_percent(usable.get('analysis_usable_rate')) if usable is not None else 'n/a'}.",
                risk_variant,
            )
        )
    if best is not None:
        cards.append(
            (
                "Recommended interpretation",
                f"Using the app-side heuristic usable rate minus normalized parse, validation, cost, and latency penalties, {display_config(best)} is the strongest operational choice among the selected systems. This is a UI heuristic for exploration, not a dissertation result.",
                "purple",
            )
        )

    selected_labels = set(df.get("config_label", pd.Series(dtype=str)).astype(str).tolist())
    if selected_labels == {"SLM1", "SLM2"}:
        cards.append(
            (
                "SLM-only comparison",
                "With only SLM1 and SLM2 selected, read the result as a compact SLM-only trade-off: cost is low for both, while parse reliability and usable output decide whether either is operationally defensible.",
                "info",
            )
        )
    if {"LLM1", "SLM-dom SLM2→LLM2"}.issubset(selected_labels):
        llm = df[df["config_label"] == "LLM1"].iloc[0]
        slm_dom = df[df["config_label"] == "SLM-dom SLM2→LLM2"].iloc[0]
        cards.append(
            (
                "LLM1 vs SLM-dominant",
                f"Against LLM1, SLM-dom SLM2→LLM2 changes cost by {delta_text(metric_value(llm, 'avg_total_cost_usd'), metric_value(slm_dom, 'avg_total_cost_usd'), 'currency')} and latency by {delta_text(metric_value(llm, 'mean_latency_ms'), metric_value(slm_dom, 'mean_latency_ms'), 'ms')}; parse reliability remains the key deployment check.",
                "success",
            )
        )
    return cards[:5]


def quality_ratio_row(df: pd.DataFrame, quality_col: str, denom_col: str) -> pd.Series | None:
    if df.empty or quality_col not in df.columns or denom_col not in df.columns:
        return None
    values = pd.to_numeric(df[quality_col], errors="coerce")
    denom = pd.to_numeric(df[denom_col], errors="coerce")
    ratio = values / denom.replace(0, np.nan)
    valid = df[ratio.notna()].copy()
    if valid.empty:
        return None
    valid["_ratio"] = ratio[ratio.notna()]
    return valid.sort_values("_ratio", ascending=False).iloc[0]


def generate_frontier_interpretation(
    df: pd.DataFrame,
    selected_configs: Sequence[str],
    selected_families: Sequence[str],
    quality_signal: str,
) -> list[tuple[str, str, str]]:
    quality_col = (
        "mean_overall_gold_quality_score"
        if quality_signal == "Gold quality"
        else "mean_overall_manual_quality"
    )
    quality_name = "gold quality" if quality_signal == "Gold quality" else "manual quality"
    quality_context = (
        "deterministic proxy scoring over gold-overlap tickets"
        if quality_signal == "Gold quality"
        else "blinded human audit judgement"
    )
    highest_quality = best_row(df, quality_col, highest=True)
    lowest_cost = best_row(df, "avg_total_cost_usd", highest=False)
    fastest = best_row(df, "mean_latency_ms", highest=False)
    parse_risk = best_row(df, "parse_error_rate", highest=True)
    validation_risk = best_row(df, "validation_error_rate", highest=True)
    q_per_dollar = quality_ratio_row(df, quality_col, "avg_total_cost_usd")
    q_per_latency = quality_ratio_row(df, quality_col, "mean_latency_ms")

    cards: list[tuple[str, str, str]] = []
    if highest_quality is not None:
        cards.append(
            (
                "Quality quadrant",
                f"Within {len(df)} selected system(s), {display_config(highest_quality)} has the highest {quality_name} ({format_number(highest_quality.get(quality_col), 3)}). {display_config(q_per_dollar)} gives the strongest quality-per-dollar signal. This uses {quality_context}.",
                "success",
            )
        )
    if fastest is not None:
        cards.append(
            (
                "Latency quadrant",
                f"{display_config(fastest)} is fastest at {format_ms(fastest.get('mean_latency_ms'))}. {display_config(q_per_latency)} has the strongest quality-per-latency signal; if quality is not competitive, speed should be treated as an operational advantage rather than a correctness guarantee.",
                "info",
            )
        )
    if parse_risk is not None:
        risk = parse_risk
        if validation_risk is not None and (
            metric_value(validation_risk, "validation_error_rate") or 0.0
        ) > (metric_value(parse_risk, "parse_error_rate") or 0.0):
            risk = validation_risk
        cards.append(
            (
                "Reliability gate",
                f"{display_config(risk)} has the highest selected parse/validation risk ({format_percent(risk.get('parse_error_rate'))} parse, {format_percent(risk.get('validation_error_rate'))} validation). It may require output-contract hardening even if it is cheap or fast.",
                "warning",
            )
        )
    if lowest_cost is not None:
        cards.append(
            (
                "Cost frontier",
                f"{display_config(lowest_cost)} is the lowest-cost selected system at {format_currency(lowest_cost.get('avg_total_cost_usd'))}. It is not automatically preferable unless the selected {quality_name} and reliability metrics remain acceptable.",
                "purple",
            )
        )
    return cards[:4]


def generate_gold_manual_interpretation(
    df: pd.DataFrame,
    selected_configs: Sequence[str],
    quality_signal: str,
) -> list[tuple[str, str, str]]:
    cards: list[tuple[str, str, str]] = []
    top_manual = best_row(df, "mean_overall_manual_quality", highest=True)
    top_gold = best_row(df, "mean_overall_gold_quality_score", highest=True)
    top_accuracy = best_row(df, "generated_category_accuracy", highest=True)
    low_cost = best_row(df, "avg_total_cost_usd", highest=False)

    if top_manual is not None:
        cards.append(
            (
                "Top manual quality",
                f"{display_config(top_manual)} leads the selected systems on blinded manual quality at {format_number(top_manual.get('mean_overall_manual_quality'), 2)}.",
                "success",
            )
        )
    if top_gold is not None:
        cards.append(
            (
                "Top deterministic gold score",
                f"{display_config(top_gold)} leads deterministic gold quality at {format_number(top_gold.get('mean_overall_gold_quality_score'), 3)}.",
                "info",
            )
        )
    if top_manual is not None and top_gold is not None:
        same = top_manual.get("analysis_config") == top_gold.get("analysis_config")
        cards.append(
            (
                "Manual vs deterministic agreement",
                (
                    f"Manual and deterministic scoring agree on {display_config(top_manual)} as the selected leader."
                    if same
                    else f"Manual and deterministic scoring diverge: manual favors {display_config(top_manual)}, while gold proxy favors {display_config(top_gold)}. This supports keeping the human audit layer."
                ),
                "success" if same else "warning",
            )
        )
    if low_cost is not None and top_manual is not None:
        cards.append(
            (
                "Low-cost acceptability",
                f"{display_config(low_cost)} is the lowest-cost selected system at {format_currency(low_cost.get('avg_total_cost_usd'))}. Compare it with {display_config(top_manual)} before treating cost savings as human-acceptable answer quality.",
                "purple",
            )
        )
    if top_accuracy is not None and top_manual is not None:
        cards.append(
            (
                "Why this matters",
                f"{display_config(top_accuracy)} has the strongest generated category accuracy in the selected subset, while {display_config(top_manual)} leads manual answer quality. If these differ, the app is showing the classification/generation distinction directly.",
                "warning",
            )
        )
    return cards[:5]


def generate_ticket_interpretation(
    ticket_rows: pd.DataFrame,
    selected_ticket_id: str,
    selected_configs: Sequence[str],
) -> list[tuple[str, str, str]]:
    cheapest = best_row(ticket_rows, "cost_usd", highest=False)
    fastest = best_row(ticket_rows, "latency_ms", highest=False)
    score_col = (
        "manual_quality_score"
        if "manual_quality_score" in ticket_rows.columns
        and pd.to_numeric(ticket_rows["manual_quality_score"], errors="coerce").notna().any()
        else "deterministic_gold_score"
    )
    best_score = best_row(ticket_rows, score_col, highest=True)
    parse_issues = (
        ticket_rows[ticket_rows["parse_error"].map(has_text)]
        if "parse_error" in ticket_rows
        else pd.DataFrame()
    )
    validation_issues = (
        ticket_rows[ticket_rows["validation_error"].map(has_text)]
        if "validation_error" in ticket_rows
        else pd.DataFrame()
    )
    categories = (
        ticket_rows.get("generated_predicted_category", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    escalations = (
        ticket_rows.get("escalated", pd.Series(dtype=str))
        .map(bool_label)
        .dropna()
        .unique()
        .tolist()
        if "escalated" in ticket_rows
        else []
    )

    cards = [
        (
            "Ticket-specific summary",
            f"Ticket {selected_ticket_id} is shown for {len(ticket_rows)} selected system(s). The interpretation uses only available cost, latency, score, category, escalation, parse, and validation fields.",
            "info",
        )
    ]
    if cheapest is not None and fastest is not None:
        cards.append(
            (
                "Cost and latency",
                f"{display_config(cheapest)} is cheapest at {format_currency(cheapest.get('cost_usd'))}; {display_config(fastest)} is fastest at {format_ms(fastest.get('latency_ms'))}.",
                "success",
            )
        )
    if best_score is not None:
        cards.append(
            (
                "Highest available score",
                f"Based on available {score_col.replace('_', ' ')} fields, {display_config(best_score)} has the highest score ({format_number(best_score.get(score_col), 3)}). This does not claim correctness beyond the recorded score.",
                "purple",
            )
        )
    issue_names = sorted(
        set(parse_issues.get("config_label", pd.Series(dtype=str)).tolist())
        | set(validation_issues.get("config_label", pd.Series(dtype=str)).tolist())
    )
    cards.append(
        (
            "Structural reliability",
            (
                f"Based on parse/validation flags, structural issues appear for: {', '.join(issue_names)}."
                if issue_names
                else "Based on parse/validation flags, no selected system shows a structural issue for this ticket."
            ),
            "warning" if issue_names else "success",
        )
    )
    cards.append(
        (
            "System disagreement",
            f"Generated categories {'differ' if len(categories) > 1 else 'agree'} across selected systems ({', '.join(categories) if categories else 'n/a'}). Escalation flags {'differ' if len(escalations) > 1 else 'do not differ'} based on available metadata.",
            "warning" if len(categories) > 1 or len(escalations) > 1 else "info",
        )
    )
    return cards[:5]


def generate_system_comparison(left: pd.Series, right: pd.Series) -> tuple[str, str, str]:
    left_name = display_config(left)
    right_name = display_config(right)
    cost_delta = delta_text(
        metric_value(left, "cost_usd"), metric_value(right, "cost_usd"), "currency"
    )
    latency_delta = delta_text(
        metric_value(left, "latency_ms"), metric_value(right, "latency_ms"), "ms"
    )
    quality_col = (
        "manual_quality_score"
        if metric_value(left, "manual_quality_score") is not None
        or metric_value(right, "manual_quality_score") is not None
        else "deterministic_gold_score"
    )
    quality_delta = delta_text(
        metric_value(left, quality_col), metric_value(right, quality_col), "score"
    )
    left_structural = has_text(left.get("parse_error")) or has_text(left.get("validation_error"))
    right_structural = has_text(right.get("parse_error")) or has_text(right.get("validation_error"))
    left_escalated = truthy(left.get("escalated"))
    right_escalated = truthy(right.get("escalated"))

    defensible = "both systems require score and flag review"
    if left_structural and not right_structural:
        defensible = f"{right_name} is more operationally defensible on structure for this ticket"
    elif right_structural and not left_structural:
        defensible = f"{left_name} is more operationally defensible on structure for this ticket"
    elif (
        metric_value(left, quality_col) is not None and metric_value(right, quality_col) is not None
    ):
        defensible = (
            f"{right_name} has the stronger available quality score"
            if (metric_value(right, quality_col) or 0) > (metric_value(left, quality_col) or 0)
            else f"{left_name} has the stronger available quality score"
        )

    return (
        "System comparison",
        f"Compared with {left_name}, {right_name} changes cost by {cost_delta}, latency by {latency_delta}, and available quality by {quality_delta}. Parse/validation differs: {left_structural} vs {right_structural}. Escalation differs: {left_escalated} vs {right_escalated}. Based on available fields, {defensible}.",
        "warning" if left_structural or right_structural else "success",
    )


def render_interpretation_cards(cards: Sequence[tuple[str, str, str]]) -> None:
    if not cards:
        render_insight_card(
            "Interpretation unavailable",
            "The selected subset does not contain enough numeric fields to compute an interpretation.",
            "warning",
        )
        return
    first_row = cards[:3]
    for card_col, (title, body, variant) in zip(st.columns(len(first_row)), first_row):
        with card_col:
            render_insight_card(title, body, variant)
    for title, body, variant in cards[3:]:
        render_insight_card(title, body, variant)


def page_executive_snapshot() -> None:
    render_hero()

    render_section_header(
        "Executive snapshot",
        "A polished static benchmark surface for the final RouterGym evidence.",
        "The page highlights the size of the experiment, the fixed memory layer, and the quality validation used to defend the conversion claim.",
    )
    render_metric_grid(
        [
            ("Generated outputs", "60,000", "Final balanced production-scale benchmark.", "60k"),
            ("Configurations", "6", "LLM-only, SLM-only, and SLM-dominant variants.", "CFG"),
            (
                "Tickets per config",
                "10,000",
                "Matched ticket IDs under a fixed memory layer.",
                "10k",
            ),
            (
                "Memory layer",
                "BM25-RAG",
                "Memory was fixed for the final router/model comparison.",
                "RAG",
            ),
            ("Gold-scored outputs", "456", "76 overlapping tickets across all six configs.", "G"),
            (
                "Manual audit",
                "Completed",
                "Blinded human review of generated support resolutions.",
                "H",
            ),
        ],
        columns=3,
    )

    comparator = load_config_comparator()
    slm_tradeoff = comparator[
        comparator["analysis_config"] == "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25"
    ]
    tradeoff_body = "The SLM-dominant SLM2→LLM2 system is the strongest operational trade-off candidate in the final benchmark."
    if not slm_tradeoff.empty:
        row = slm_tradeoff.iloc[0]
        tradeoff_body = (
            f"{row['config_label_full']} combines {format_currency(row.get('avg_total_cost_usd'))} "
            f"average cost, {format_ms(row.get('mean_latency_ms'))} mean latency, "
            f"{format_percent(row.get('analysis_usable_rate'))} usable outputs, and "
            f"{format_percent(row.get('parse_error_rate'))} parse errors."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_insight_card(
            "What this proves",
            "RouterGym moves beyond model preference charts: it measures conversion feasibility through cost, latency, parse reliability, gold scoring, and blinded human audit.",
            "success",
        )
    with col2:
        render_insight_card("Best operational trade-off", tradeoff_body, "purple")
    with col3:
        render_insight_card(
            "Important caveat",
            "SLM-dominant routing is a conditional conversion strategy, not a universal replacement claim. Quality and structural reliability remain gating constraints.",
            "warning",
        )

    if not comparator.empty:
        render_section_header(
            "Operational preview",
            "Cost, latency, and reliability at a glance.",
            "These preview charts use the committed summary CSVs; no model inference is executed.",
        )
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            render_chart_card(
                "Average cost per ticket",
                make_horizontal_bar(
                    comparator, "avg_total_cost_usd", x_title="USD per ticket", height=360
                ),
                "Lower is better, but only if usable output and parse reliability remain acceptable.",
            )
        with chart_col2:
            render_chart_card(
                "Usable output rate",
                make_horizontal_bar(
                    comparator,
                    "analysis_usable_rate",
                    x_title="Usable output rate",
                    tickformat=".1%",
                    height=360,
                ),
                "A cheap system that fails structural checks is not production-ready.",
            )

    render_static_plot_card(
        "Committed benchmark figure",
        [
            "slm_dominant_vs_llm_baseline_tradeoff.png",
            PLOTS_DIR / "slm_dominant_vs_llm_baseline_tradeoff.png",
        ],
        "PNG discovery searches analysis outputs, plot folders, manual audit, and gold-resolution directories.",
    )


def page_configuration_comparator() -> None:
    render_section_header(
        "Configuration comparator",
        "Operational dashboard for the six final benchmark configurations.",
        "Filter router families and compare cost, latency, usable output, and parse reliability without changing any benchmark values.",
    )
    df = load_config_comparator()
    if df.empty:
        render_insight_card("Missing summary", "Could not load `summary_by_config.csv`.", "warning")
        return

    df, selected_routers, selected_configs = family_filter_row(df, key_prefix="comparator")
    if df.empty:
        render_insight_card(
            "No rows selected", "No configurations match the current filters.", "warning"
        )
        return

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_chart_card(
            "Cost per ticket",
            make_horizontal_bar(df, "avg_total_cost_usd", x_title="USD per ticket"),
            "Token-cost telemetry under normalized pricing assumptions.",
        )
    with chart_col2:
        render_chart_card(
            "Mean latency",
            make_horizontal_bar(df, "mean_latency_ms", x_title="Milliseconds"),
            "Observed mean latency from the final benchmark run.",
        )

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        render_chart_card(
            "Usable output rate",
            make_horizontal_bar(df, "analysis_usable_rate", x_title="Rate", tickformat=".1%"),
            "Share of rows that passed structural checks for downstream use.",
        )
    with chart_col4:
        render_chart_card(
            "Parse error rate",
            make_horizontal_bar(df, "parse_error_rate", x_title="Rate", tickformat=".1%"),
            "Parse failures remain operational failures even when text looks plausible.",
        )

    render_interpretation_cards(
        generate_configuration_interpretation(df, selected_configs, selected_routers)
    )

    with st.expander("Formatted comparison table", expanded=False):
        with st.container(border=True):
            st.dataframe(comparison_table(df), hide_index=True, **stretch_width())


def page_quality_evaluation() -> None:
    render_section_header(
        "Gold and manual quality",
        "Quality is treated as a layered validation problem.",
        "Classification accuracy is not generated-answer correctness. Deterministic gold scoring is a proxy, and the blinded manual audit is the human validation layer.",
    )

    gold = standardize_config_labels(
        safe_read_csv(GOLD_DIR / "gold_resolution_quality_by_config.csv")
    )
    manual = standardize_config_labels(safe_read_csv(MANUAL_DIR / "manual_quality_by_config.csv"))
    components = standardize_config_labels(
        safe_read_csv(MANUAL_DIR / "manual_quality_component_summary_by_config.csv")
    )
    manual_frontier = standardize_config_labels(
        safe_read_csv(MANUAL_DIR / "manual_quality_vs_cost_latency.csv")
    )
    comparator = load_config_comparator()

    quality_summary = comparator[
        [
            column
            for column in [
                "analysis_config",
                "config_label",
                "config_label_full",
                "router_family_label",
                "_config_rank",
                "avg_total_cost_usd",
                "mean_latency_ms",
                "analysis_usable_rate",
                "parse_error_rate",
                "validation_error_rate",
            ]
            if column in comparator.columns
        ]
    ].copy()
    if quality_summary.empty and not gold.empty:
        quality_summary = gold[
            [
                column
                for column in [
                    "analysis_config",
                    "config_label",
                    "config_label_full",
                    "router_family_label",
                    "_config_rank",
                ]
                if column in gold.columns
            ]
        ].copy()
    for source, columns in [
        (
            manual,
            [
                "analysis_config",
                "mean_overall_manual_quality",
                "manual_quality_pass_rate_7",
            ],
        ),
        (
            gold,
            [
                "analysis_config",
                "mean_overall_gold_quality_score",
                "generated_category_accuracy",
                "pass_rate_070",
                "pass_rate_080",
            ],
        ),
    ]:
        if not source.empty and not quality_summary.empty:
            keep = [column for column in columns if column in source.columns]
            quality_summary = quality_summary.merge(source[keep], on="analysis_config", how="left")

    with st.container(border=True):
        st.markdown("**Quality interpretation controls**")
        col1, col2 = st.columns([1.6, 1])
        config_options = (
            quality_summary.sort_values("_config_rank")["config_label"].drop_duplicates().tolist()
            if not quality_summary.empty
            else []
        )
        with col1:
            selected_configs = st.multiselect(
                "Configurations",
                config_options,
                default=config_options,
                key="quality_configs",
            )
        with col2:
            quality_focus = st.selectbox(
                "Quality focus",
                ["Combined", "Manual quality", "Gold quality"],
                key="quality_focus",
            )

    def _filter_quality(source: pd.DataFrame) -> pd.DataFrame:
        if source.empty or not selected_configs:
            return source
        return source[source["config_label"].isin(selected_configs)].reset_index(drop=True)

    gold = _filter_quality(gold)
    manual = _filter_quality(manual)
    components = _filter_quality(components)
    manual_frontier = _filter_quality(manual_frontier)
    quality_summary = _filter_quality(quality_summary)

    render_interpretation_cards(
        generate_gold_manual_interpretation(quality_summary, selected_configs, quality_focus)
    )

    gold_tab, manual_tab, component_tab, frontier_tab = st.tabs(
        ["Gold Quality", "Manual Audit", "Components", "Quality Frontier"]
    )

    with gold_tab:
        render_insight_card(
            "Deterministic gold scoring",
            "Gold quality compares generated resolutions with frozen reference resolutions. It is a reproducible proxy, not a substitute for human judgment.",
            "info",
        )
        if gold.empty:
            render_insight_card(
                "Missing gold summary", "Gold quality CSV is unavailable.", "warning"
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                render_chart_card(
                    "Overall gold quality",
                    make_horizontal_bar(
                        gold, "mean_overall_gold_quality_score", x_title="Mean score"
                    ),
                    "Deterministic generated-answer quality on the 456-output gold overlap.",
                )
            with col2:
                render_chart_card(
                    "Generated category accuracy",
                    make_horizontal_bar(
                        gold, "generated_category_accuracy", x_title="Accuracy", tickformat=".1%"
                    ),
                    "This is category-label agreement only, not answer correctness.",
                )
            with st.expander("Gold quality table", expanded=False):
                st.dataframe(gold, hide_index=True, **stretch_width())

    with manual_tab:
        render_insight_card(
            "Blinded manual audit",
            "The human audit scores category understanding, actionability, completeness, resolution steps, escalation appropriateness, and policy grounding.",
            "success",
        )
        if manual.empty:
            render_insight_card(
                "Missing manual summary", "Manual audit summary CSV is unavailable.", "warning"
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                render_chart_card(
                    "Mean manual quality",
                    make_horizontal_bar(
                        manual, "mean_overall_manual_quality", x_title="Mean score"
                    ),
                    "Human-scored generated resolution quality.",
                )
            with col2:
                render_chart_card(
                    "Manual pass rate >= 7",
                    make_horizontal_bar(
                        manual, "manual_quality_pass_rate_7", x_title="Pass rate", tickformat=".1%"
                    ),
                    "Share of audited outputs meeting the manual quality threshold.",
                )
            with st.expander("Manual audit table", expanded=False):
                st.dataframe(manual, hide_index=True, **stretch_width())

    with component_tab:
        render_insight_card(
            "Component-level diagnosis",
            "The component view shows whether a system fails because answers are incomplete, steps are weak, escalation is poor, or policy grounding is thin.",
            "purple",
        )
        if components.empty:
            render_insight_card(
                "Missing component summary", "Manual component CSV is unavailable.", "warning"
            )
        else:
            component_cols = [
                column
                for column in components.columns
                if column.startswith("mean_") and column.endswith("_manual")
            ]
            if not component_cols:
                render_insight_card(
                    "Missing component columns",
                    "No manual component score columns were found.",
                    "warning",
                )
            else:
                component_df = components.melt(
                    id_vars=[
                        "analysis_config",
                        "config_label_full",
                        "router_family_label",
                        "_config_rank",
                    ],
                    value_vars=component_cols,
                    var_name="Component",
                    value_name="Mean score",
                )
                component_df["Component"] = (
                    component_df["Component"]
                    .str.replace("mean_", "", regex=False)
                    .str.replace("_manual", "", regex=False)
                    .str.replace("_", " ")
                    .str.title()
                )
                component_df = component_df.sort_values("_config_rank", ascending=False)
                fig = px.bar(
                    component_df,
                    x="Mean score",
                    y="config_label_full",
                    color="Component",
                    barmode="group",
                    orientation="h",
                    color_discrete_sequence=[
                        "#243b63",
                        "#138a72",
                        "#d5001f",
                        "#6c4ad2",
                        "#6f7d91",
                        "#a45f13",
                    ],
                    hover_data=["analysis_config", "Component"],
                )
                fig.update_layout(**chart_layout(640), bargap=0.18)
                fig.update_xaxes(
                    title="Mean manual component score",
                    showgrid=True,
                    gridcolor="rgba(23,32,51,0.08)",
                )
                fig.update_yaxes(title="", automargin=True)
                render_chart_card(
                    "Manual component scores",
                    fig,
                    "Grouped horizontal bars keep the long configuration labels readable.",
                )
                with st.expander("Component score table", expanded=False):
                    st.dataframe(components, hide_index=True, **stretch_width())

    with frontier_tab:
        render_insight_card(
            "Quality frontier",
            "Frontier charts make the conversion trade-off explicit: lower cost or latency is only attractive when quality remains defensible.",
            "info",
        )
        if manual_frontier.empty:
            render_insight_card(
                "Missing frontier data",
                "Manual quality cost/latency CSV is unavailable.",
                "warning",
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                render_chart_card(
                    "Manual quality vs cost",
                    make_scatter(
                        manual_frontier,
                        "avg_cost_usd",
                        "mean_overall_manual_quality",
                        x_title="Average cost per ticket",
                        y_title="Mean manual quality",
                        size="manual_quality_pass_rate_7",
                    ),
                    "Point size reflects manual pass rate.",
                )
            with col2:
                render_chart_card(
                    "Manual quality vs latency",
                    make_scatter(
                        manual_frontier,
                        "avg_latency_ms",
                        "mean_overall_manual_quality",
                        x_title="Average latency (ms)",
                        y_title="Mean manual quality",
                        size="manual_quality_pass_rate_7",
                    ),
                    "Lower latency is only useful if answer quality holds.",
                )


def page_frontier() -> None:
    render_section_header(
        "Cost-latency-quality frontier",
        "A visual frontier for deciding whether conversion is operationally defensible.",
        "Lower cost plus higher quality is better. Lower latency plus higher quality is better. Parse reliability can disqualify otherwise cheap systems.",
    )
    frontier = load_quality_frontier()
    if frontier.empty:
        render_insight_card(
            "Missing frontier data", "Required summaries are not available.", "warning"
        )
        return

    with st.container(border=True):
        st.markdown("**Frontier controls**")
        col1, col2, col3 = st.columns([1, 1.5, 1])
        router_options = sorted(frontier["router_family_label"].dropna().unique().tolist())
        config_options = (
            frontier.sort_values("_config_rank")["config_label"].drop_duplicates().tolist()
        )
        with col1:
            selected_routers = st.multiselect(
                "Router family",
                router_options,
                default=router_options,
                key="frontier_routers",
            )
        with col2:
            selected_configs = st.multiselect(
                "Configuration",
                config_options,
                default=config_options,
                key="frontier_configs",
            )
        with col3:
            quality_options = []
            if (
                "mean_overall_manual_quality" in frontier.columns
                and frontier["mean_overall_manual_quality"].notna().any()
            ):
                quality_options.append("Manual quality")
            if (
                "mean_overall_gold_quality_score" in frontier.columns
                and frontier["mean_overall_gold_quality_score"].notna().any()
            ):
                quality_options.append("Gold quality")
            selected_quality = st.selectbox("Quality signal", quality_options or ["Unavailable"])

    frontier = filtered_configs(frontier, selected_routers, selected_configs)
    if frontier.empty:
        render_insight_card(
            "No rows selected", "No configurations match the current filters.", "warning"
        )
        return

    quality_col = (
        "mean_overall_gold_quality_score"
        if selected_quality == "Gold quality"
        else "mean_overall_manual_quality"
    )
    quality_title = (
        "Mean gold quality" if selected_quality == "Gold quality" else "Mean manual quality"
    )

    render_interpretation_cards(
        generate_frontier_interpretation(
            frontier,
            selected_configs,
            selected_routers,
            selected_quality,
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        render_chart_card(
            f"{selected_quality} vs cost",
            make_scatter(
                frontier,
                "avg_total_cost_usd",
                quality_col,
                x_title="Average cost per ticket",
                y_title=quality_title,
                size="analysis_usable_rate",
            ),
            "Point size reflects usable output rate.",
        )
    with col2:
        render_chart_card(
            f"{selected_quality} vs latency",
            make_scatter(
                frontier,
                "mean_latency_ms",
                quality_col,
                x_title="Mean latency (ms)",
                y_title=quality_title,
                size="analysis_usable_rate",
            ),
            "Point size reflects usable output rate.",
        )

    col3, col4 = st.columns(2)
    with col3:
        render_chart_card(
            "Reliability bubble chart",
            make_scatter(
                frontier,
                "parse_error_rate",
                quality_col,
                x_title="Parse error rate",
                y_title=quality_title,
                size="analysis_usable_rate",
                x_tickformat=".1%",
            ),
            "Lower parse-error rate and higher quality is preferable.",
        )
    with col4:
        render_chart_card(
            "Parse error vs usable output",
            make_scatter(
                frontier,
                "parse_error_rate",
                "analysis_usable_rate",
                x_title="Parse error rate",
                y_title="Usable output rate",
                size=quality_col,
                x_tickformat=".1%",
                y_tickformat=".1%",
            ),
            "A system can be cheap but operationally weak if usability drops.",
        )


def status_badges(row: pd.Series) -> str:
    parse_ok = not has_text(row.get("parse_error"))
    validation_ok = not has_text(row.get("validation_error"))
    escalated = truthy(row.get("escalated"))
    usable = truthy(row.get("usable_output"))
    return " ".join(
        [
            render_badge("parse OK" if parse_ok else "parse error", "green" if parse_ok else "red"),
            render_badge(
                "validation OK" if validation_ok else "validation error",
                "green" if validation_ok else "red",
            ),
            render_badge(
                "escalated" if escalated else "not escalated", "purple" if escalated else "blue"
            ),
            render_badge("usable" if usable else "not usable", "green" if usable else "warning"),
        ]
    )


def render_answer_header(row: pd.Series, display_name: str, blind_mode: bool) -> None:
    if blind_mode:
        subtitle = "Blinded benchmark system"
    else:
        escalation = str(row.get("escalation_model_key", ""))
        suffix = f" → {escalation}" if has_text(escalation) and escalation.lower() != "none" else ""
        subtitle = (
            f"{row.get('router_family_label', '')} | "
            f"{row.get('base_model_key', '')}{suffix} | "
            f"{row.get('memory_mode', '')}"
        )
    st.markdown(
        f"""
<div class="rg-answer-card">
    <div class="rg-answer-head">
        <div>
            <h3>{escape(display_name)}</h3>
            <div class="rg-answer-subtitle">{escape(subtitle)}</div>
        </div>
        <div class="rg-answer-badges">{status_badges(row)}</div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def ticket_system_map(configs: Sequence[str]) -> dict[str, str]:
    ordered = [config for config in CONFIG_ORDER if config in configs]
    ordered.extend(config for config in configs if config not in ordered)
    return {config: f"System {chr(65 + index)}" for index, config in enumerate(ordered)}


def page_ticket_explorer() -> None:
    render_section_header(
        "Ticket explorer",
        "Concrete benchmark rows without a long-scroll wall of text.",
        "Use filters, inspect one ticket, compare generated answers, and optionally blind system identities.",
    )
    sample = standardize_config_labels(safe_read_csv(SAMPLE_PATH))
    if sample.empty:
        render_insight_card(
            "Missing ticket sample",
            "Run `python RouterGym\\analysis\\build_streamlit_ticket_sample.py` to create the deployable sample.",
            "warning",
        )
        return

    sample["ticket_id"] = sample["ticket_id"].astype(str)
    category_col = first_existing_column(sample, ["gold_label", "topic_group"])
    all_config_ids = (
        sample.sort_values("_config_rank")["analysis_config"].drop_duplicates().tolist()
    )
    blind_map = ticket_system_map(all_config_ids)

    with st.container(border=True):
        st.markdown("**Ticket filters**")
        col0, col1, col2, col3 = st.columns([0.8, 1.1, 1.2, 1.8])
        with col0:
            blind_mode = st.checkbox("Blind mode", value=False)
        with col1:
            categories = ["All"]
            if category_col:
                categories += sorted(sample[category_col].dropna().astype(str).unique().tolist())
            selected_category = st.selectbox("Category", categories)
        with col2:
            routers = sorted(sample["router_family_label"].dropna().astype(str).unique().tolist())
            selected_routers = st.multiselect("Router family", routers, default=routers)
        with col3:
            config_options = sample[
                ["analysis_config", "config_label", "_config_rank"]
            ].drop_duplicates()
            config_options = config_options.sort_values("_config_rank")
            label_for_config = {
                row["analysis_config"]: (
                    blind_map[row["analysis_config"]] if blind_mode else row["config_label"]
                )
                for _, row in config_options.iterrows()
            }
            reverse_config = {label: config for config, label in label_for_config.items()}
            config_labels = list(reverse_config.keys())
            selected_config_labels = st.multiselect(
                "Systems" if blind_mode else "Configurations",
                config_labels,
                default=config_labels,
            )
            selected_config_ids = [reverse_config[label] for label in selected_config_labels]

    filtered = sample.copy()
    if selected_category != "All" and category_col:
        filtered = filtered[filtered[category_col].astype(str) == selected_category]
    if selected_routers:
        filtered = filtered[filtered["router_family_label"].isin(selected_routers)]
    if selected_config_ids:
        filtered = filtered[filtered["analysis_config"].isin(selected_config_ids)]

    if filtered.empty:
        render_insight_card("No rows selected", "No rows match the current filters.", "warning")
        return

    ticket_options = (
        filtered[["ticket_id", category_col if category_col else "ticket_id"]]
        .drop_duplicates()
        .sort_values(
            "ticket_id",
            key=lambda series: pd.to_numeric(series, errors="coerce").fillna(10**9),
        )
    )
    labels: list[str] = []
    label_to_id: dict[str, str] = {}
    for _, row in ticket_options.iterrows():
        label = (
            f"{row['ticket_id']} - {row[category_col]}" if category_col else str(row["ticket_id"])
        )
        labels.append(label)
        label_to_id[label] = str(row["ticket_id"])

    selected_ticket_label = st.selectbox("Ticket ID", labels)
    selected_ticket_id = label_to_id[selected_ticket_label]
    ticket_rows = filtered[filtered["ticket_id"] == selected_ticket_id].sort_values(
        ["_config_rank", "config_label"]
    )
    first_row = ticket_rows.iloc[0]

    visible_configs = ticket_rows["analysis_config"].drop_duplicates().tolist()
    local_blind_map = ticket_system_map(visible_configs)

    def display_name(row: pd.Series) -> str:
        if blind_mode:
            return local_blind_map.get(row["analysis_config"], "System")
        return str(row["config_label_full"])

    interpretation_rows = ticket_rows.copy()
    interpretation_rows["config_label"] = interpretation_rows.apply(display_name, axis=1)
    render_interpretation_cards(
        generate_ticket_interpretation(
            interpretation_rows,
            selected_ticket_id,
            selected_config_ids,
        )
    )

    ticket_tab, gold_tab, generated_tab, compare_tab, scores_tab = st.tabs(
        [
            "Ticket",
            "Gold Reference",
            "Generated Answers",
            "Compare Two Systems",
            "Scores & Reliability",
        ]
    )

    with ticket_tab:
        render_metric_grid(
            [
                ("Ticket ID", selected_ticket_id, "Curated deployable sample ticket.", "ID"),
                (
                    "Gold label",
                    str(first_row.get("gold_label", "n/a")),
                    "Reference category.",
                    "CAT",
                ),
                ("Systems shown", str(len(ticket_rows)), "Rows after active filters.", "SYS"),
            ],
            columns=3,
        )
        render_insight_card(
            "Ticket metadata",
            f"Sample source: {first_row.get('sample_source', 'n/a')}. Gold review status: {first_row.get('gold_review_status', 'n/a')}.",
            "info",
        )
        with st.container(border=True):
            st.markdown("#### Ticket text")
            st.write(first_row.get("ticket_text", first_row.get("original_query", "")))

    with gold_tab:
        if has_text(first_row.get("gold_resolution_summary")):
            render_insight_card(
                "Gold reference summary", str(first_row.get("gold_resolution_summary")), "info"
            )
        else:
            render_insight_card(
                "Gold reference summary", "No gold summary is available for this ticket.", "warning"
            )
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("#### Expected steps")
                steps = parse_list_value(first_row.get("gold_resolution_steps"))
                if steps:
                    for step in steps:
                        st.markdown(f"- {step}")
                else:
                    st.caption("No expected steps available.")
        with col2:
            with st.container(border=True):
                st.markdown("#### Acceptance criteria")
                criteria = parse_list_value(first_row.get("gold_acceptance_criteria"))
                if criteria:
                    for item in criteria:
                        st.markdown(f"- {item}")
                else:
                    st.caption("No acceptance criteria available.")

    with generated_tab:
        for _, row in ticket_rows.iterrows():
            render_answer_header(row, display_name(row), blind_mode)
            render_metric_grid(
                [
                    (
                        "Generated category",
                        str(row.get("generated_predicted_category", "n/a")),
                        "Model-assigned output category.",
                        "CAT",
                    ),
                    (
                        "Escalated",
                        bool_label(row.get("escalated")),
                        "Whether the row used escalation.",
                        "ESC",
                    ),
                    ("Latency", format_ms(row.get("latency_ms")), "Observed row latency.", "LAT"),
                    ("Cost", format_currency(row.get("cost_usd")), "Observed row cost.", "$"),
                    (
                        "Gold score",
                        format_number(row.get("deterministic_gold_score"), 3),
                        "Deterministic gold proxy.",
                        "G",
                    ),
                    (
                        "Manual score",
                        format_number(row.get("manual_quality_score"), 1),
                        "Blinded audit score if available.",
                        "H",
                    ),
                ],
                columns=3,
            )
            with st.expander("Generated answer", expanded=False):
                st.write(row.get("final_answer", ""))
            steps = parse_list_value(row.get("resolution_steps"))
            if steps:
                with st.expander("Resolution steps", expanded=False):
                    for step in steps:
                        st.markdown(f"- {step}")
            if has_text(row.get("reasoning")):
                with st.expander("Reasoning", expanded=False):
                    st.write(row.get("reasoning"))

    with compare_tab:
        options = [display_name(row) for _, row in ticket_rows.iterrows()]
        if len(options) < 2:
            render_insight_card(
                "Comparison unavailable", "Select at least two systems to compare.", "warning"
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                left_name = st.selectbox("System A", options, index=0)
            with col2:
                right_name = st.selectbox("System B", options, index=1)
            lookup = {display_name(row): row for _, row in ticket_rows.iterrows()}
            left_compare = lookup[left_name].copy()
            right_compare = lookup[right_name].copy()
            left_compare["config_label"] = left_name
            right_compare["config_label"] = right_name
            render_interpretation_cards([generate_system_comparison(left_compare, right_compare)])
            for column, name in zip(st.columns(2), [left_name, right_name]):
                row = lookup[name]
                with column:
                    render_answer_header(row, name, blind_mode)
                    render_metric_grid(
                        [
                            ("Cost", format_currency(row.get("cost_usd")), "Row cost.", "$"),
                            ("Latency", format_ms(row.get("latency_ms")), "Row latency.", "LAT"),
                            (
                                "Parse",
                                "OK" if not has_text(row.get("parse_error")) else "Error",
                                "Parser status.",
                                "P",
                            ),
                            (
                                "Validation",
                                "OK" if not has_text(row.get("validation_error")) else "Error",
                                "Schema status.",
                                "V",
                            ),
                            (
                                "Gold score",
                                format_number(row.get("deterministic_gold_score"), 3),
                                "Deterministic proxy.",
                                "G",
                            ),
                            (
                                "Manual score",
                                format_number(row.get("manual_quality_score"), 1),
                                "Human audit.",
                                "H",
                            ),
                        ],
                        columns=2,
                    )
                    with st.expander("Answer", expanded=True):
                        st.write(row.get("final_answer", ""))
                    steps = parse_list_value(row.get("resolution_steps"))
                    if steps:
                        with st.expander("Steps", expanded=False):
                            for step in steps:
                                st.markdown(f"- {step}")

    with scores_tab:
        score_table = ticket_rows.copy()
        score_table["System"] = score_table.apply(display_name, axis=1)
        score_columns = [
            "System",
            "generated_predicted_category",
            "generation_valid",
            "usable_output",
            "parse_error",
            "validation_error",
            "cost_usd",
            "latency_ms",
            "total_tokens",
            "deterministic_gold_score",
            "manual_quality_score",
            "manual_answer_actionable",
            "manual_answer_complete",
            "manual_policy_grounded",
        ]
        st.dataframe(
            score_table[[column for column in score_columns if column in score_table.columns]],
            hide_index=True,
            **stretch_width(),
        )


def render_html_grid(class_name: str, cards: Sequence[tuple[str, str, str]]) -> None:
    card_class = class_name.replace("-grid", "-card")
    items = "\n".join(
        f"""
<article class="{card_class}">
    <span>{escape(kicker)}</span>
    <strong>{escape(title)}</strong>
    <p style="margin:0.45rem 0 0;color:#667085;line-height:1.45;font-size:0.84rem;">{escape(body)}</p>
</article>
        """
        for kicker, title, body in cards
    )
    st.markdown(f'<div class="{class_name}">{items}</div>', unsafe_allow_html=True)


def page_methodology_static_archive() -> None:
    render_section_header(
        "Methodology and reproducibility",
        "The benchmark story as an auditable static research product.",
        "This page summarizes the pipeline, inference environment, persistent storage, and reproducibility guarantees behind the demo.",
    )

    render_section_header("Pipeline", "Ticket-to-audit architecture")
    pipeline_cards = [
        ("Step 1", "Ticket", "A normalized support ticket enters the benchmark."),
        ("Step 2", "Classifier", "The classifier predicts the operational category."),
        ("Step 3", "Router", "The router selects LLM-only, SLM-only, or SLM-dominant execution."),
        ("Step 4", "BM25-RAG", "The fixed memory layer retrieves policy context."),
        (
            "Step 5",
            "Model / Escalation",
            "The model generates the answer or escalates selectively.",
        ),
        ("Step 6", "JSON Validation", "Structured output is parsed and validated."),
        (
            "Step 7",
            "Scoring / Audit",
            "Operational, gold, and manual quality signals are produced.",
        ),
    ]
    render_html_grid("rg-pipeline-grid", pipeline_cards)

    render_section_header("Experimental matrix", "Final six-configuration BM25-RAG comparison")
    matrix = pd.DataFrame(
        [
            ("LLM-only", "LLM1", "none", "BM25-RAG"),
            ("LLM-only", "LLM2", "none", "BM25-RAG"),
            ("SLM-only", "SLM1", "none", "BM25-RAG"),
            ("SLM-only", "SLM2", "none", "BM25-RAG"),
            ("SLM-dominant", "SLM1", "LLM2", "BM25-RAG"),
            ("SLM-dominant", "SLM2", "LLM2", "BM25-RAG"),
        ],
        columns=["Router family", "Base model", "Escalation model", "Memory"],
    )
    st.dataframe(matrix, hide_index=True, **stretch_width())
    render_insight_card(
        "Why BM25-RAG was fixed",
        "The final production-scale comparison isolates router/model strategy under a consistent operational memory layer. Full memory ablation is future work for this claim.",
        "info",
    )

    render_section_header(
        "Inference environment", "Final run infrastructure and projected full-run economics"
    )
    render_metric_grid(
        [
            ("RunPod GPUs", "4 × H200 SXM", "Production inference hardware.", "GPU"),
            ("GPU rate", "$2.99/GPU-hour", "Rental rate used for estimates.", "$"),
            ("Runtime", "24 hours", "Final 60k benchmark run.", "24h"),
            ("Final run cost", "$287.04", "Estimated final GPU run cost.", "$"),
            ("Projected outputs", "287,022", "47,837 tickets across six configurations.", "OUT"),
            ("Projected full run", "$1,373", "Estimated full-run cost on the same setup.", "EST"),
        ],
        columns=3,
    )

    render_section_header("Persistent storage", "Run artifacts kept outside the static app")
    storage_cards = [
        ("Storage", "model cache", "Model and tokenizer cache on the RunPod volume."),
        ("Code", "RouterGym project", "Repository snapshot and benchmark scripts."),
        ("Serving", "vLLM logs", "Local serving logs and runtime traces."),
        ("Outputs", "chunked outputs", "Chunked result files written during production execution."),
        ("Control", "manifests/status", "Status files used to track progress and recovery."),
        ("Recovery", "recovered results", "Recovered and merged intermediate outputs."),
        ("Bundles", "final bundles", "Final summaries used by this static demo."),
    ]
    render_html_grid("rg-storage-grid", storage_cards)

    render_section_header("Reproducibility checklist", "Static demo safeguards")
    checklist = [
        (
            "Static",
            "committed summaries",
            "The app reads committed summary CSVs and compact samples.",
        ),
        (
            "Offline",
            "no live inference",
            "No GPUs, servers, or model calls are started by the app.",
        ),
        ("Secrets", "no API keys", "The UI does not require provider credentials."),
        ("Models", "no downloads", "The app does not download or load model weights."),
        ("Git", "raw data excluded", "Large raw analysis inputs stay out of GitHub."),
        (
            "Backup",
            "raw archive backup",
            "Raw archive is backed up separately outside the repository.",
        ),
        (
            "Audit",
            "manual audit included",
            "Human audit summaries are committed as static outputs.",
        ),
        (
            "Evidence",
            "gold scoring included",
            "Deterministic gold-resolution summaries are committed.",
        ),
    ]
    check_items = "\n".join(
        f"""
<article class="rg-check-item">
    <span>{escape(kicker)}</span>
    <strong>{escape(title)}</strong>
    <p style="margin:0.45rem 0 0;color:#667085;line-height:1.45;font-size:0.84rem;">{escape(body)}</p>
</article>
        """
        for kicker, title, body in checklist
    )
    st.markdown(f'<div class="rg-check-grid">{check_items}</div>', unsafe_allow_html=True)

    render_section_header("Run locally", "Streamlit Cloud-compatible launch command")
    st.code("streamlit run streamlit_app.py", language="powershell")


def render_flow_track(stages: Sequence[str], selected_stage: str) -> None:
    items = "\n".join(
        f"""
<article class="rg-flow-stage {"rg-flow-stage-active" if stage == selected_stage else ""}">
    <span>Stage {index}</span>
    <strong>{escape(stage)}</strong>
</article>
        """
        for index, stage in enumerate(stages, start=1)
    )
    st.markdown(f'<div class="rg-flow-track">{items}</div>', unsafe_allow_html=True)


def page_methodology() -> None:
    render_section_header(
        "Methodology and reproducibility",
        "The benchmark story as an auditable static research product.",
        "Select a pipeline stage to inspect what happens, why it matters, what artefacts are produced, and which caveats keep the static demo defensible.",
    )

    stage_details = {
        "Ticket input": {
            "what": "A normalized support ticket enters the benchmark with ticket text and a reference topic label.",
            "why": "It anchors every generated output to a concrete support request and keeps comparisons matched across configurations.",
            "artefacts": "ticket_id, ticket_text, gold_label, topic_group",
            "section": "Dataset construction and benchmark design",
            "metric": "RouterGym/data/tickets/tickets.csv and streamlit_ticket_sample.csv",
            "caution": "The app uses a compact committed sample for inspection, not the raw full analysis table.",
        },
        "Classifier": {
            "what": "The classifier predicts ticket category and confidence metadata before answer generation.",
            "why": "It separates the routing/classification layer from generated answer quality.",
            "artefacts": "predicted_category, classifier metadata, confidence fields where available",
            "section": "Routing and classification layer",
            "metric": "classification_metrics_by_config.csv",
            "caution": "Classifier accuracy is not final answer correctness.",
        },
        "Router": {
            "what": "The router selects LLM-only, SLM-only, or SLM-dominant execution with optional escalation.",
            "why": "This is the conversion test from LLM-first operation toward SLM-dominant systems.",
            "artefacts": "analysis_config, router_family, escalation_flags, escalated",
            "section": "Router policy and conversion strategy",
            "metric": "routing_escalation_summary.csv",
            "caution": "Routing benefits must be read with reliability and quality signals.",
        },
        "BM25-RAG memory": {
            "what": "A lexical BM25 retrieval layer supplies policy/support context to each final configuration.",
            "why": "Fixing the memory layer controls the retrieval condition in the final model/router comparison.",
            "artefacts": "memory_mode, retrieved policy context metadata where available",
            "section": "Memory layer and retrieval control",
            "metric": "summary_by_config.csv",
            "caution": "The final claim is not a full memory-mode ablation.",
        },
        "Model generation": {
            "what": "The selected open-weight model generates a structured support resolution, or escalates where the route requires it.",
            "why": "Final answer quality, cost, and latency are measured at this layer.",
            "artefacts": "final_answer, resolution_steps, reasoning, cost_usd, latency_ms",
            "section": "Generation and inference execution",
            "metric": "generation_quality_by_config.csv and latency_summary_by_config.csv",
            "caution": "The Streamlit app does not run live inference or download models.",
        },
        "JSON validation": {
            "what": "Generated outputs are parsed and checked against the expected output contract.",
            "why": "Structurally invalid output can fail automation even if the natural-language text looks useful.",
            "artefacts": "parse_error, validation_error, generation_valid, usable_output",
            "section": "Output contract and reliability",
            "metric": "reliability_by_config.csv and summary_by_config.csv",
            "caution": "Parse reliability can disqualify otherwise cheap or fast systems.",
        },
        "Result recovery": {
            "what": "Chunked outputs are recovered, merged, audited for available rows, and balanced for comparison.",
            "why": "This protects the integrity of the 60k final analysis dataset.",
            "artefacts": "chunked outputs, manifests/status files, recovered results, final bundles",
            "section": "Production execution and recovery",
            "metric": "dataset_integrity_report.json and dataset_integrity_by_config.csv",
            "caution": "Raw large outputs remain excluded from GitHub and are not loaded by the app.",
        },
        "Gold scoring": {
            "what": "Generated answers on overlapping gold tickets are scored against deterministic reference resolutions.",
            "why": "This supplies a reproducible quality proxy for the 456-output gold overlap.",
            "artefacts": "overall_gold_quality_score, component gold scores, generated_category_accuracy",
            "section": "Deterministic quality scoring",
            "metric": "gold_resolution_quality_by_config.csv",
            "caution": "Gold scoring is useful but is not full human judgement.",
        },
        "Manual audit": {
            "what": "A blinded human review scores answer quality, actionability, completeness, escalation, and policy grounding.",
            "why": "It validates whether deterministic and operational metrics align with human judgement.",
            "artefacts": "overall_manual_quality, component manual scores, reviewer notes",
            "section": "Human validation layer",
            "metric": "manual_quality_by_config.csv and manual_quality_component_summary_by_config.csv",
            "caution": "Manual audit is the human validation layer, not a live production approval system.",
        },
        "Final analysis": {
            "what": "Committed summary CSVs and plots power the static Streamlit explorer.",
            "why": "The demo remains deployable, auditable, and safe without credentials or heavy runtime dependencies.",
            "artefacts": "summary_by_config.csv, plots, manual/gold summaries, streamlit_ticket_sample.csv",
            "section": "Results synthesis and reproducibility",
            "metric": "RouterGym/results/analysis_outputs/",
            "caution": "No external APIs, live inference, model downloads, or raw 1.27GB data are used.",
        },
    }

    stages = list(stage_details)
    selected_stage = st.radio(
        "Pipeline stage",
        stages,
        horizontal=True,
        label_visibility="collapsed",
        key="methodology_stage",
    )
    render_flow_track(stages, selected_stage)
    details = stage_details[selected_stage]

    col1, col2, col3 = st.columns(3)
    with col1:
        render_insight_card("What happens", details["what"], "info")
    with col2:
        render_insight_card("Why it matters", details["why"], "success")
    with col3:
        render_insight_card("Caution", details["caution"], "warning")

    render_metric_grid(
        [
            ("Artefacts", details["artefacts"], "Produced or consumed at this stage.", "ART"),
            (
                "Dissertation section",
                details["section"],
                "Where the stage fits in the story.",
                "SEC",
            ),
            ("Metric or file", details["metric"], "Static artefact used by the app.", "FILE"),
        ],
        columns=3,
    )

    render_section_header("Experimental matrix", "Final six-configuration BM25-RAG comparison")
    matrix = pd.DataFrame(
        [
            ("LLM-only", "LLM1", "none", "BM25-RAG"),
            ("LLM-only", "LLM2", "none", "BM25-RAG"),
            ("SLM-only", "SLM1", "none", "BM25-RAG"),
            ("SLM-only", "SLM2", "none", "BM25-RAG"),
            ("SLM-dominant", "SLM1", "LLM2", "BM25-RAG"),
            ("SLM-dominant", "SLM2", "LLM2", "BM25-RAG"),
        ],
        columns=["Router family", "Base model", "Escalation model", "Memory"],
    )
    st.dataframe(matrix, hide_index=True, **stretch_width())
    render_insight_card(
        "Why BM25-RAG was fixed",
        "The final production-scale comparison isolates router/model strategy under a consistent operational memory layer. Full memory ablation is future work for this claim.",
        "info",
    )

    render_section_header("Experiment calculator", "Projected static-run economics")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            tickets = st.number_input("Tickets", min_value=1, value=47_837, step=100)
            configs = st.number_input("Configurations", min_value=1, value=6, step=1)
        with col2:
            gpu_count = st.number_input("GPU count", min_value=1, value=4, step=1)
            gpu_rate = st.number_input("GPU hourly rate", min_value=0.0, value=2.99, step=0.10)
        with col3:
            observed_outputs = st.number_input(
                "Observed outputs", min_value=1, value=60_000, step=1_000
            )
            observed_runtime = st.number_input(
                "Observed 60k runtime (hours)", min_value=0.1, value=24.0, step=0.5
            )
    projected_outputs = int(tickets * configs)
    projected_runtime = projected_outputs * (observed_runtime / observed_outputs)
    projected_cost = projected_runtime * gpu_count * gpu_rate
    render_metric_grid(
        [
            ("Projected outputs", f"{projected_outputs:,.0f}", "tickets x configurations.", "OUT"),
            (
                "Projected runtime",
                f"{projected_runtime:,.1f} h",
                "Scaled from observed runtime/output ratio.",
                "TIME",
            ),
            (
                "Projected GPU rental",
                f"${projected_cost:,.0f}",
                "runtime x GPU count x hourly rate.",
                "$",
            ),
        ],
        columns=3,
    )

    render_section_header("Persistent storage", "Run artefacts kept outside the static app")
    storage_cards = [
        ("Storage", "model cache", "Model and tokenizer cache on the RunPod volume."),
        ("Code", "RouterGym project", "Repository snapshot and benchmark scripts."),
        ("Serving", "vLLM logs", "Local serving logs and runtime traces."),
        ("Outputs", "chunked outputs", "Chunked result files written during production execution."),
        ("Control", "manifests/status", "Status files used to track progress and recovery."),
        ("Recovery", "recovered results", "Recovered and merged intermediate outputs."),
        ("Bundles", "final bundles", "Final summaries used by this static demo."),
    ]
    render_html_grid("rg-storage-grid", storage_cards)

    render_section_header("Reproducibility checklist", "Static demo safeguards")
    checklist = [
        (
            "Static",
            "committed summaries",
            "The app reads committed summary CSVs and compact samples.",
        ),
        (
            "Offline",
            "no live inference",
            "No GPUs, servers, or model calls are started by the app.",
        ),
        ("Secrets", "no API keys", "The UI does not require provider credentials."),
        ("Models", "no downloads", "The app does not download or load model weights."),
        ("Git", "raw data excluded", "Large raw analysis inputs stay out of GitHub."),
        (
            "Backup",
            "raw archive backup",
            "Raw archive is backed up separately outside the repository.",
        ),
        (
            "Audit",
            "manual audit included",
            "Human audit summaries are committed as static outputs.",
        ),
        (
            "Evidence",
            "gold scoring included",
            "Deterministic gold-resolution summaries are committed.",
        ),
    ]
    check_items = "\n".join(
        f"""
<article class="rg-check-item">
    <span>{escape(kicker)}</span>
    <strong>{escape(title)}</strong>
    <p style="margin:0.45rem 0 0;color:var(--muted);line-height:1.45;font-size:0.84rem;">{escape(body)}</p>
</article>
        """
        for kicker, title, body in checklist
    )
    st.markdown(f'<div class="rg-check-grid">{check_items}</div>', unsafe_allow_html=True)

    render_section_header("Run locally", "Streamlit Cloud-compatible launch command")
    st.code("streamlit run streamlit_app.py", language="powershell")


PAGE_RENDERERS = {
    "Executive Snapshot": page_executive_snapshot,
    "Configuration Comparator": page_configuration_comparator,
    "Gold and Manual Quality": page_quality_evaluation,
    "Cost-Latency-Quality Frontier": page_frontier,
    "Ticket Explorer": page_ticket_explorer,
    "Methodology and Reproducibility": page_methodology,
}


def main() -> None:
    st.session_state.setdefault("dark_mode", False)
    inject_global_css(st.session_state["dark_mode"])
    page = render_top_nav("Executive Snapshot")
    PAGE_RENDERERS[page]()


if __name__ == "__main__":
    main()
