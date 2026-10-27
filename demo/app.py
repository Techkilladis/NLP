"""
DA-RAG Studio: Dual-Method Code Generation & Context Retrieval Test Bench.
A research-grade web application comparing Baseline (None / Zero Retrieval)
versus DA-RAG (Dependency Graph Retrieval) powered by Qwen2.5-Coder-1.5B-Instruct.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd

import data_loader
import evaluator
import dag_visualizer
import interactive_runner

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DA-RAG Studio · Dual-Method Code Generation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global Design System & CSS ──────────────────────────────────────────────
# Implements modern web development best practices:
# 1. `dark-mode`: color-scheme: dark on :root, html, body ensures native form controls
#    and scrollbars adhere to dark theme.
# 2. `color`: WCAG AA/AAA compliant contrast ratios (>4.5:1 text, >3:1 UI borders).
#    State is conveyed with iconography + text badges, never color alone.
# 3. Streamlit enclosure: bulletproof CSS overrides that eliminate light-theme leaks.
st.markdown("""
<style>
/* ── 1. DESIGN TOKENS ──────────────────────────────────────────────────────── */
:root {
    color-scheme: dark !important;

    /* Backgrounds */
    --canvas: #07090E;
    --surface-1: #0D121F;
    --surface-2: #131A2B;
    --surface-3: #1A233A;

    /* Borders */
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-mid: rgba(255, 255, 255, 0.14);
    --border-focus: #6366F1;

    /* Accents */
    --brand-500: #6366F1;
    --brand-400: #818CF8;
    --brand-glow: rgba(99, 102, 241, 0.22);

    /* Semantic DAG (Emerald) */
    --dag-500: #10B981;
    --dag-400: #34D399;
    --dag-bg: rgba(16, 185, 129, 0.1);
    --dag-glow: rgba(16, 185, 129, 0.2);

    /* Semantic None (Rose/Red) */
    --none-500: #EF4444;
    --none-400: #F87171;
    --none-bg: rgba(239, 68, 68, 0.1);
    --none-glow: rgba(239, 68, 68, 0.18);

    /* Text */
    --text-pure: #FFFFFF;
    --text-primary: #F8FAFC;
    --text-secondary: #CBD5E1;
    --text-muted: #94A3B8;

    /* Typography */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

    /* Scrollbars (dark-mode guide) */
    --scrollbar-track: #0B0F19;
    --scrollbar-thumb: #334155;
    scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
    scrollbar-width: thin;
}

/* ── 2. STREAMLIT APP ENCLOSURE (FORCES TOTAL DARK MODE) ───────────────────── */
html, body {
    background-color: var(--canvas) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    margin: 0;
    padding: 0;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stBottom"],
[data-testid="stToolbar"],
.main,
.block-container,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background-color: var(--canvas) !important;
    color: var(--text-primary) !important;
}

/* Remove default Streamlit top header whitespace */
header[data-testid="stHeader"] {
    background: var(--canvas) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    height: 3.2rem !important;
}

/* Main container layout */
.main .block-container {
    padding: 1.5rem 3rem 4rem !important;
    max-width: 1440px !important;
}

/* Modern typography import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: var(--font-sans) !important;
    -webkit-font-smoothing: antialiased;
}
code, pre, .stCode * {
    font-family: var(--font-mono) !important;
}

/* ── 3. TOP APP BAR ────────────────────────────────────────────────────────── */
.app-header-strip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 20px;
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.app-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-logo-badge {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 16px var(--brand-glow);
}
.brand-name {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-pure);
}
.brand-sub {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(99, 102, 241, 0.15);
    color: var(--brand-400);
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.telemetry-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.telemetry-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface-2);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
}
.telemetry-chip strong {
    color: var(--text-primary);
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
    display: inline-block;
}

/* ── 4. HERO SECTION ───────────────────────────────────────────────────────── */
.hero-box {
    margin-bottom: 24px;
    padding: 8px 4px;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--brand-400);
    margin-bottom: 8px;
}
.hero-heading {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 60%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-lead {
    font-size: 1.0rem;
    color: var(--text-secondary);
    max-width: 820px;
    line-height: 1.6;
}

/* ── 5. STREAMLIT WIDGETS OVERRIDES ────────────────────────────────────────── */
/* Tabs as modern segmented pill control */
div[data-baseweb="tab-list"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 6px !important;
    width: fit-content !important;
    margin-bottom: 24px !important;
}
div[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.15s ease-in-out !important;
}
div[data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(255, 255, 255, 0.04) !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: var(--surface-3) !important;
    color: var(--text-pure) !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}
div[data-baseweb="tab-border"],
div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: var(--surface-1) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
    background-color: var(--surface-1) !important;
    border: none !important;
    color: var(--text-primary) !important;
}
div[data-baseweb="select"] * {
    color: var(--text-primary) !important;
}
div[data-baseweb="popover"],
ul[role="listbox"],
li[role="option"] {
    background-color: var(--surface-2) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-subtle) !important;
}
li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: var(--surface-3) !important;
    color: #60A5FA !important;
}

/* Text Area */
div[data-baseweb="textarea"],
div[data-baseweb="textarea"] > div,
textarea {
    background-color: #0A0E17 !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
}
textarea:focus {
    border-color: var(--brand-500) !important;
    box-shadow: 0 0 0 2px var(--brand-glow) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.6rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 4px 16px var(--brand-glow) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.45) !important;
    background: linear-gradient(135deg, #7175F3 0%, #5850EC 100%) !important;
}

/* Code blocks */
div[data-testid="stCode"],
.stCode > div {
    background: #0A0E17 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
}

/* Expanders */
details[data-testid="stExpander"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
}
details[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Checkbox */
div[data-testid="stCheckbox"] label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ── 6. COMPONENT CARDS ────────────────────────────────────────────────────── */

/* Insight Callout */
.insight-deck {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.insight-icon {
    font-size: 20px;
    line-height: 1;
}
.insight-title {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--brand-400);
    margin-bottom: 4px;
}
.insight-text {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.55;
}

/* Verdict Banner */
.verdict-box {
    position: relative;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 26px;
    border: 1.5px solid;
}
.verdict-dag-win {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.25) 100%);
    border-color: var(--dag-500);
    box-shadow: 0 0 35px var(--dag-glow);
}
.verdict-none-win {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(127, 29, 29, 0.25) 100%);
    border-color: var(--none-500);
    box-shadow: 0 0 35px var(--none-glow);
}
.verdict-tie-win {
    background: var(--surface-1);
    border-color: var(--border-mid);
}
.verdict-header-row {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.verdict-lead {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text-pure);
    letter-spacing: -0.02em;
}
.verdict-pill {
    font-size: 0.85rem;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
}
.verdict-pill-dag {
    background: var(--dag-bg);
    color: var(--dag-400);
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.verdict-pill-none {
    background: var(--none-bg);
    color: var(--none-400);
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.verdict-pill-tie {
    background: rgba(245, 158, 11, 0.12);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.verdict-explanation {
    font-size: 0.92rem;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 960px;
}

/* Stage Heading Strip */
.stage-strip {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 32px 0 14px;
}
.stage-strip::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
}
.stage-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

/* Dual Column Comparison Card */
.comparison-card {
    background: var(--surface-1);
    border: 1.5px solid var(--border-subtle);
    border-radius: 14px;
    overflow: hidden;
    height: 100%;
}
.comparison-card-dag {
    border-color: rgba(16, 185, 129, 0.45);
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.08);
}
.comparison-card-none {
    border-color: rgba(239, 68, 68, 0.3);
}
.comparison-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1px solid var(--border-subtle);
}
.comparison-header-dag {
    background: rgba(16, 185, 129, 0.08);
}
.comparison-header-none {
    background: rgba(239, 68, 68, 0.06);
}
.comparison-header-title {
    font-size: 0.88rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
}
.comparison-header-dag .comparison-header-title {
    color: var(--dag-400);
}
.comparison-header-none .comparison-header-title {
    color: var(--none-400);
}

.token-chip {
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    font-family: var(--font-mono);
}
.token-chip-dag {
    background: var(--dag-bg);
    color: var(--dag-400);
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.token-chip-none {
    background: var(--none-bg);
    color: var(--none-400);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Empty State Card */
.zero-context-void {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 45px 24px;
    text-align: center;
    border: 1.5px dashed rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    margin: 16px;
    background: rgba(0, 0, 0, 0.2);
}
.zero-context-icon {
    font-size: 32px;
    margin-bottom: 10px;
}
.zero-context-heading {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 6px;
}
.zero-context-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    max-width: 360px;
    line-height: 1.5;
}

/* Generation Telemetry Strip */
.gen-telemetry {
    display: flex;
    gap: 16px;
    padding: 8px 18px;
    background: var(--surface-2);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
}
.gen-telemetry strong {
    color: var(--text-secondary);
}

/* Status Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
}
.status-pill-pass {
    background: var(--dag-bg);
    color: var(--dag-400);
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.status-pill-fail {
    background: var(--none-bg);
    color: var(--none-400);
    border: 1px solid rgba(239, 68, 68, 0.4);
}

/* Scorecard Grid */
.score-deck {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.score-deck-cell {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.score-deck-cell-dag {
    border-color: rgba(16, 185, 129, 0.45);
    box-shadow: 0 0 25px var(--dag-glow);
}
.score-cell-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.score-cell-number {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 6px;
    font-family: var(--font-sans);
}
.score-cell-number-dag { color: var(--dag-400); }
.score-cell-number-none { color: var(--none-400); }
.score-cell-number-delta { color: var(--brand-400); }
.score-cell-sub {
    font-size: 0.76rem;
    color: var(--text-muted);
}

/* Progress Bars (semantic and accessible per `dark-mode` & `color` guides) */
.dim-grid-box {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 16px 20px;
}
.dim-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid var(--border-subtle);
}
.dim-row:last-child {
    border-bottom: none;
}
.dim-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
    min-width: 260px;
}
.dim-bars {
    flex: 1;
    max-width: 600px;
}
.dim-single-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.dim-tag {
    font-size: 0.7rem;
    font-weight: 700;
    width: 38px;
    text-align: right;
    font-family: var(--font-mono);
}
.dim-tag-none { color: var(--none-400); }
.dim-tag-dag  { color: var(--dag-400); }
.dim-val {
    font-size: 0.75rem;
    font-weight: 700;
    width: 40px;
    font-family: var(--font-mono);
}

progress.modern-bar {
    appearance: none;
    -webkit-appearance: none;
    border: 0;
    height: 8px;
    flex: 1;
    border-radius: 9999px;
    background: var(--surface-3);
    overflow: hidden;
}
progress.modern-bar::-webkit-progress-bar { background: transparent; }
progress.modern-bar.bar-none::-webkit-progress-value {
    background: var(--none-500);
    border-radius: 9999px;
}
progress.modern-bar.bar-dag::-webkit-progress-value {
    background: var(--dag-500);
    border-radius: 9999px;
}
progress.modern-bar.bar-none::-moz-progress-bar {
    background: var(--none-500);
    border-radius: 9999px;
}
progress.modern-bar.bar-dag::-moz-progress-bar {
    background: var(--dag-500);
    border-radius: 9999px;
}

/* Footer */
.app-footer {
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.footer-text {
    font-size: 0.8rem;
    color: var(--text-muted);
}
.footer-tag {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 6px;
    background: var(--dag-bg);
    color: var(--dag-400);
    border: 1px solid rgba(16, 185, 129, 0.3);
}
</style>
""", unsafe_allow_html=True)


# ─── Data Caching ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading benchmark metadata…")
def get_cached_metadata():
    tasks = data_loader.load_all_task_metadata()
    eligibility = data_loader.load_eligibility_sets()
    pilot_summary = data_loader.load_pilot_summary_csv()
    return tasks, eligibility, pilot_summary


@st.cache_data(show_spinner="Loading verified predictions…")
def get_cached_records(run_type: str):
    run_dir = data_loader.PILOT_GEN_DIR if run_type == "pilot" else data_loader.FULL_GEN_DIR
    preds, pred_inc = data_loader.load_run_predictions(run_dir)
    prompts, prompt_inc = data_loader.load_run_prompts(run_dir)
    eval_recs, eval_inc = data_loader.load_run_functional_eval(run_dir)
    return preds, prompts, eval_recs, (pred_inc or prompt_inc or eval_inc)


all_tasks, eligibility_sets, pilot_summary = get_cached_metadata()
eligible_ids = eligibility_sets["eligible"]
ineligible_ids = eligibility_sets["ineligible"]
pilot_preds, pilot_prompts, pilot_evals, _ = get_cached_records("pilot")


# ─── Top Application Bar ─────────────────────────────────────────────────────
st.markdown("""
<div class="app-header-strip">
  <div class="app-brand">
    <div class="brand-logo-badge">⚡</div>
    <div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span class="brand-name">DA-RAG Studio</span>
        <span class="brand-sub">RESEARCH BENCHMARK</span>
      </div>
    </div>
  </div>
  <div class="telemetry-group">
    <span class="telemetry-chip"><span class="pulse-dot"></span> <strong>Qwen2.5-Coder-1.5B</strong></span>
    <span class="telemetry-chip">⚡ <strong>NVIDIA RTX 4050 (CUDA)</strong></span>
    <span class="telemetry-chip">📦 <strong>2,048 Tokens</strong> Budget</span>
    <span class="telemetry-chip">🏆 Pilot: <strong>8 Passes vs 4 (+100%)</strong></span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Hero Overview ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <div class="hero-tag">🔬 Dual-Method Code Generation Benchmark</div>
  <div class="hero-heading">Observe the Visible Difference: Baseline vs DA-RAG</div>
  <div class="hero-lead">
    Zero-Retrieval LLMs frequently hallucinate or misuse APIs on repository-specific functions.
    DA-RAG extracts AST call graphs and packs transitively reachable definitions under a strict 2,048-token knapsack budget.
    Run live GPU inference or inspect verified completions side-by-side.
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Navigation Tabs ─────────────────────────────────────────────────────────
tab_play, tab_bench, tab_method = st.tabs([
    "⚡  Interactive Playground",
    "📊  DevEval Benchmarks (1,496 Tasks)",
    "🔬  Scientific Methodology & DAG Architecture",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INTERACTIVE PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════
with tab_play:
    st.markdown('<div class="stage-strip">1 — Benchmark Setup & Input</div>', unsafe_allow_html=True)

    ctrl_col, insight_col = st.columns([1.1, 1.5], gap="large")

    preset_options = ["✍️  Custom Function / Prompt Input"] + [p["title"] for p in interactive_runner.DEMO_PRESETS]

    with ctrl_col:
        selected_mode = st.selectbox(
            "Select Benchmark Exemplar or Custom Input",
            preset_options,
            index=2,  # Default to FS: File Wildcard Matcher
            help="Choose an exemplar where DA-RAG demonstrates clear superiority, or write custom Python code.",
        )
        run_live_gpu = st.checkbox(
            "⚡ Run Live GPU Inference (Qwen2.5-Coder on CUDA)",
            value=False,
            help="Unchecked: Loads verified instant pre-computed completions. Checked: Executes fresh live inference on your RTX 4050 GPU.",
        )

    active_preset = next(
        (p for p in interactive_runner.DEMO_PRESETS if p["title"] == selected_mode),
        None
    )
    default_code = (
        active_preset["sample_input"]
        if active_preset
        else "def parse_query_params(url: str):\n    '''Parse URL query string and return a dict of unique parameters.'''\n"
    )

    with insight_col:
        if active_preset:
            st.markdown(f"""
            <div class="insight-deck">
              <div class="insight-icon">💡</div>
              <div>
                <div class="insight-title">Scientific Insight · Why DA-RAG Wins Here</div>
                <div class="insight-text">{active_preset['why_dag_wins']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-deck">
              <div class="insight-icon">✍️</div>
              <div>
                <div class="insight-title">Custom Function Mode</div>
                <div class="insight-text">
                  Provide any Python function signature and docstring. The interactive runner will extract the function name,
                  query the repository dependency graph, pack reachable symbols under 2,048 tokens, and generate completions for both methods.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    input_code_val = st.text_area(
        "Python Function Signature & Requirement Prompt",
        value=default_code,
        height=125,
        help="Specify the target function signature and parameter requirements.",
    )

    run_btn = st.button("🚀  Run Dual Context Retrieval · Qwen Generation · Evaluation", type="primary")

    should_run = run_btn or "last_result" not in st.session_state or (
        active_preset and st.session_state.get("last_preset") != selected_mode
    )

    if should_run:
        with st.spinner("Executing context retrieval and code generation pipelines…"):
            task_id = active_preset["id"] if active_preset else None
            result = interactive_runner.run_dual_pipeline(
                task_id=task_id,
                custom_code=input_code_val,
                run_live_gpu=run_live_gpu,
                pilot_preds=pilot_preds,
                pilot_prompts=pilot_prompts,
                pilot_evals=pilot_evals,
                pilot_summary=pilot_summary,
                all_tasks=all_tasks,
            )
            st.session_state["last_result"] = result
            st.session_state["last_preset"] = selected_mode

    result = st.session_state.get("last_result")

    if result:
        eval_data = result["evaluation"]
        res_none  = result["none"]
        res_dag   = result["dag"]
        m_none    = res_none["metrics"]
        m_dag     = res_dag["metrics"]

        # ── Winner Verdict ───────────────────────────────────────────────────
        winner = eval_data["winner"]
        if winner == "dag":
            v_cls, p_cls = "verdict-dag-win", "verdict-pill-dag"
        elif winner == "none":
            v_cls, p_cls = "verdict-none-win", "verdict-pill-none"
        else:
            v_cls, p_cls = "verdict-tie-win", "verdict-pill-tie"

        st.markdown(f"""
        <div class="verdict-box {v_cls}">
          <div class="verdict-header-row">
            <span class="verdict-lead">{eval_data['winner_label']}</span>
            <span class="verdict-pill {p_cls}">{eval_data['delta_str']}</span>
            <span style="font-size:0.85rem; color:var(--text-muted); font-family:var(--font-mono);">
              DAG Score: {eval_data['score_dag']}% · Baseline Score: {eval_data['score_none']}%
            </span>
          </div>
          <div class="verdict-explanation">{eval_data['narrative']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 1: Context Retrieval ──────────────────────────────────────
        st.markdown('<div class="stage-strip">Stage 1 — Context Retrieval Visualizer</div>', unsafe_allow_html=True)
        st.markdown('<div class="stage-title">Visible Difference in Injected Context</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--text-muted); font-size:0.86rem; margin-bottom:16px;">The model receives identical instructions; the sole independent variable is the repository context injected into the prompt.</p>', unsafe_allow_html=True)

        ctx_col1, ctx_col2 = st.columns(2, gap="medium")

        with ctx_col1:
            st.markdown("""
            <div class="comparison-card comparison-card-none">
              <div class="comparison-header comparison-header-none">
                <div class="comparison-header-title">🚫 Baseline · Zero Retrieval</div>
                <span class="token-chip token-chip-none">0 / 2,048 tok (0% budget)</span>
              </div>
              <div style="padding: 12px 16px;">
                <div class="zero-context-void">
                  <div class="zero-context-icon">⬛</div>
                  <div class="zero-context-heading">Zero Repository Context</div>
                  <div class="zero-context-sub">
                    The prompt contains only the isolated function signature and docstring.
                    No internal helpers, constants, or exception classes are supplied.
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("🔍 View Raw Baseline Prompt Sent to Qwen", expanded=False):
                st.code(res_none["prompt_text"], language="text")

        with ctx_col2:
            dag_tok = res_dag["context_tokens"]
            n_units = res_dag["n_context_units"]

            svg_html = dag_visualizer.render_dag_graph_svg(
                target_name=result["target_name"],
                packed_items=res_dag["packed_items"],
                tokens=dag_tok,
                n_units=n_units,
            )
            st.markdown(svg_html, unsafe_allow_html=True)

            if res_dag["context_text"]:
                with st.expander(f"📦 View Injected Repository Code Context ({dag_tok} tokens)", expanded=True):
                    st.code(res_dag["context_text"], language="python")
            if res_dag["packed_items"]:
                with st.expander(f"📑 Packed Graph Units Table ({n_units} units)", expanded=False):
                    try:
                        st.dataframe(pd.DataFrame(res_dag["packed_items"]), width="stretch")
                    except Exception:
                        st.dataframe(pd.DataFrame(res_dag["packed_items"]))
            with st.expander("🔍 View Raw DA-RAG Prompt Sent to Qwen", expanded=False):
                st.code(res_dag["prompt_text"], language="text")

        # ── Section 2: Generated Code ─────────────────────────────────────────
        st.markdown('<div class="stage-strip">Stage 2 — Qwen2.5-Coder Generation</div>', unsafe_allow_html=True)
        st.markdown('<div class="stage-title">Side-by-Side Model Completions</div>', unsafe_allow_html=True)

        gen_col1, gen_col2 = st.columns(2, gap="medium")

        n_passed = res_none.get("eval_rec") and res_none["eval_rec"].get("official_result") == "Pass"
        d_passed = res_dag.get("eval_rec") and res_dag["eval_rec"].get("official_result") == "Pass"

        badge_none = '<span class="status-pill status-pill-pass">✓ Passed Functional Tests</span>' if n_passed else '<span class="status-pill status-pill-fail">✗ Failed Functional Tests</span>'
        badge_dag = '<span class="status-pill status-pill-pass">✓ Passed Functional Tests</span>' if d_passed else '<span class="status-pill status-pill-fail">✗ Failed Functional Tests</span>'

        with gen_col1:
            st.markdown(f"""
            <div class="comparison-header comparison-header-none" style="border-radius:12px 12px 0 0; border:1px solid rgba(239,68,68,0.3); border-bottom:none;">
              <span class="comparison-header-title">Baseline Output (none)</span>
              {badge_none}
            </div>
            <div class="gen-telemetry" style="border-left:1px solid rgba(239,68,68,0.3); border-right:1px solid rgba(239,68,68,0.3);">
              <span>TOKENS: <strong>{res_none['tokens']}</strong></span>
              <span>LATENCY: <strong>{res_none['latency']:.2f}s</strong></span>
              <span>SYNTAX: <strong>{m_none['syntax_msg']}</strong></span>
            </div>
            """, unsafe_allow_html=True)
            st.code(res_none["completion"] or "# Empty completion", language="python")
            if res_none.get("raw_output") and res_none["raw_output"] != res_none["completion"]:
                with st.expander("View Raw LLM Generation Stream", expanded=False):
                    st.code(res_none["raw_output"], language="text")

        with gen_col2:
            st.markdown(f"""
            <div class="comparison-header comparison-header-dag" style="border-radius:12px 12px 0 0; border:1px solid rgba(16,185,129,0.4); border-bottom:none;">
              <span class="comparison-header-title">DA-RAG Output (darag_b)</span>
              {badge_dag}
            </div>
            <div class="gen-telemetry" style="border-left:1px solid rgba(16,185,129,0.4); border-right:1px solid rgba(16,185,129,0.4);">
              <span>TOKENS: <strong>{res_dag['tokens']}</strong></span>
              <span>LATENCY: <strong>{res_dag['latency']:.2f}s</strong></span>
              <span>CTX TOKENS: <strong>{dag_tok}</strong></span>
              <span>SYNTAX: <strong>{m_dag['syntax_msg']}</strong></span>
            </div>
            """, unsafe_allow_html=True)
            st.code(res_dag["completion"] or "# Empty completion", language="python")
            if res_dag.get("raw_output") and res_dag["raw_output"] != res_dag["completion"]:
                with st.expander("View Raw LLM Generation Stream", expanded=False):
                    st.code(res_dag["raw_output"], language="text")

        if result.get("ground_truth"):
            with st.expander("📖 Ground-Truth Reference Implementation (from Repository)", expanded=False):
                st.code(result["ground_truth"], language="python")

        # ── Section 3: Evaluation Scorecard ──────────────────────────────────
        st.markdown('<div class="stage-strip">Stage 3 — Evaluation Scorecard</div>', unsafe_allow_html=True)
        st.markdown('<div class="stage-title">Multi-Dimensional Scoring Matrix</div>', unsafe_allow_html=True)

        score_none = eval_data["score_none"]
        score_dag  = eval_data["score_dag"]
        delta      = eval_data["delta"]
        delta_str  = f"+{delta}%" if delta > 0 else f"{delta}%"

        st.markdown(f"""
        <div class="score-deck">
          <div class="score-deck-cell">
            <div class="score-cell-label">Baseline · Zero Retrieval</div>
            <div class="score-cell-number score-cell-number-none">{score_none}<span style="font-size:1.4rem">%</span></div>
            <div class="score-cell-sub">Isolated prompt evaluation</div>
          </div>
          <div class="score-deck-cell score-deck-cell-dag">
            <div class="score-cell-label">DA-RAG · Dependency Graph</div>
            <div class="score-cell-number score-cell-number-dag">{score_dag}<span style="font-size:1.4rem">%</span></div>
            <div class="score-cell-sub">Graph-augmented prompt evaluation</div>
          </div>
          <div class="score-deck-cell">
            <div class="score-cell-label">Performance Advantage</div>
            <div class="score-cell-number score-cell-number-delta">{delta_str}</div>
            <div class="score-cell-sub">Reliability gain from context</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        dims = [
            ("Repository & Dependency Utilization", "dependency_score"),
            ("Ground-Truth Similarity",             "similarity_score"),
            ("Python AST Syntax Validity",          "syntax_score"),
            ("Interface & Contract Conformance",    "interface_score"),
        ]
        dim_rows = ""
        for label, key in dims:
            vn = m_none.get(key, 0)
            vd = m_dag.get(key, 0)
            dim_rows += f"""
            <div class="dim-row">
              <div class="dim-name">{label}</div>
              <div class="dim-bars">
                <div class="dim-single-bar">
                  <span class="dim-tag dim-tag-none">None</span>
                  <progress class="modern-bar bar-none" value="{vn}" max="100" aria-label="{label} baseline score"></progress>
                  <span class="dim-val" style="color:var(--none-400);">{vn}%</span>
                </div>
                <div class="dim-single-bar">
                  <span class="dim-tag dim-tag-dag">DAG</span>
                  <progress class="modern-bar bar-dag" value="{vd}" max="100" aria-label="{label} DA-RAG score"></progress>
                  <span class="dim-val" style="color:var(--dag-400);">{vd}%</span>
                </div>
              </div>
            </div>"""

        st.markdown(f"""
        <div class="dim-grid-box">
          {dim_rows}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BENCHMARK EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown('<div class="stage-strip">DevEval Benchmark Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">Official Cohorts & Empirical Results</div>', unsafe_allow_html=True)

    b_ctrl, b_info = st.columns([1, 1.8], gap="large")

    with b_ctrl:
        run_cohort = st.radio(
            "Benchmark Cohort",
            ["Pilot Cohort (50 tasks — official evaluation complete)", "Full Benchmark (1,496 tasks — in progress)"],
            index=0,
        )
        is_pilot = "Pilot" in run_cohort
        active_preds, active_prompts, active_evals, _ = get_cached_records("pilot" if is_pilot else "full")

        t_ids = (
            sorted({t for (t, m) in active_preds.keys() if m in ("none", "darag_b")})
            or sorted({t for (t, m) in pilot_summary.keys()})
        ) if is_pilot else sorted(eligible_ids)

        filter_b = st.selectbox(
            "Filter Task Set",
            [
                "⭐  DA-RAG Passed & Baseline Failed (5 key tasks)",
                "✅  DA-RAG Passed — All 8 pilot passes",
                "All Tasks",
                "🎯  Eligible Only",
                "⚠️  Ineligible (Error Analysis)",
            ],
            index=0 if is_pilot else 2,
        )

    # Filter task IDs
    filtered_ids = []
    for tid in t_ids:
        is_el = tid in eligible_ids
        d_p = (active_evals.get((tid, "darag_b")) or pilot_summary.get((tid, "darag_b")) or {}).get("official_result") == "Pass"
        n_f = (active_evals.get((tid, "none")) or pilot_summary.get((tid, "none")) or {}).get("official_result") != "Pass"
        if "DAG Passed & Baseline" in filter_b and d_p and n_f:
            filtered_ids.append(tid)
        elif "All 8 pilot" in filter_b and d_p:
            filtered_ids.append(tid)
        elif "All Tasks" in filter_b:
            filtered_ids.append(tid)
        elif "Eligible" in filter_b and "Ineligible" not in filter_b and is_el:
            filtered_ids.append(tid)
        elif "Ineligible" in filter_b and not is_el:
            filtered_ids.append(tid)

    if not filtered_ids:
        filtered_ids = t_ids

    with b_info:
        selected_bench_task = st.selectbox(
            f"Select Task ({len(filtered_ids)} matching)",
            filtered_ids,
            index=0,
        )
        t_data = all_tasks.get(selected_bench_task, {})
        is_t_el = selected_bench_task in eligible_ids
        elig_badge = '<span class="status-pill status-pill-pass">✓ Eligible</span>' if is_t_el else '<span class="status-pill status-pill-fail">⚠ Ineligible</span>'
        st.markdown(f"""
        <div style="margin-top: 8px; padding: 12px 16px; background: var(--surface-1); border: 1px solid var(--border-subtle); border-radius: 10px;">
          <code style="font-size:0.85rem; color: var(--brand-400);">{selected_bench_task}</code><br>
          <span style="font-size:0.78rem; color: var(--text-muted);">{t_data.get('project_path','')}</span>
          &nbsp; {elig_badge}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    res_bench = interactive_runner.run_dual_pipeline(
        task_id=selected_bench_task,
        run_live_gpu=False,
        pilot_preds=active_preds,
        pilot_prompts=active_prompts,
        pilot_evals=active_evals,
        pilot_summary=pilot_summary,
        all_tasks=all_tasks,
    )
    b_eval = res_bench["evaluation"]

    bw = b_eval["winner"]
    if bw == "dag":
        b_v_cls, b_p_cls = "verdict-dag-win", "verdict-pill-dag"
    elif bw == "none":
        b_v_cls, b_p_cls = "verdict-none-win", "verdict-pill-none"
    else:
        b_v_cls, b_p_cls = "verdict-tie-win", "verdict-pill-tie"

    st.markdown(f"""
    <div class="verdict-box {b_v_cls}">
      <div class="verdict-header-row">
        <span class="verdict-lead">{b_eval['winner_label']}</span>
        <span class="verdict-pill {b_p_cls}">{b_eval['delta_str']}</span>
        <span style="font-size:0.85rem; color:var(--text-muted); font-family:var(--font-mono);">
          DAG: {b_eval['score_dag']}% · None: {b_eval['score_none']}%
        </span>
      </div>
      <div class="verdict-explanation">{b_eval['narrative']}</div>
    </div>
    """, unsafe_allow_html=True)

    bc1, bc2 = st.columns(2, gap="medium")
    with bc1:
        st.markdown('<div style="font-size:0.82rem; font-weight:700; color:var(--none-400); margin-bottom:6px; letter-spacing:0.05em; text-transform:uppercase;">Baseline Completion (none)</div>', unsafe_allow_html=True)
        st.code(res_bench["none"]["completion"] or "# No completion recorded", language="python")
    with bc2:
        st.markdown('<div style="font-size:0.82rem; font-weight:700; color:var(--dag-400); margin-bottom:6px; letter-spacing:0.05em; text-transform:uppercase;">DA-RAG Completion (darag_b)</div>', unsafe_allow_html=True)
        st.code(res_bench["dag"]["completion"] or "# No completion recorded", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown('<div class="stage-strip">Architecture & Research Formulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">Why Dependency Graphs Transform Code Retrieval</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.markdown("""
        ### The Zero-Retrieval Hallucination Problem
        Standard LLM code generation (**Baseline `none`**) operates on isolated prompt strings. When real-world code relies on repository-specific contracts:
        - **Internal Sentinel Values**: e.g., `values.unset`
        - **Project-Internal Helpers**: e.g., `fs.wildcard.imatch`, `_parse_iso8601`
        - **Custom Exception Hierarchies**: e.g., `FSPathError`, `ConfigException`

        Without repository context, the model hallucinates generic Python standard library calls (like `fnmatch.fnmatch` or `urllib.parse.parse_qs`) that fail edge-case tests.
        """)

    with m2:
        st.markdown("""
        ### The DA-RAG Solution: Graph Traversal + Knapsack Packing
        DA-RAG formulates retrieval as a **graph-directed Multi-Choice Knapsack Problem (MCKP)**:

        1. **AST Dependency Graph Extraction**: Per-repository call graphs, import dependencies, and type hierarchies.
        2. **Multi-Hop Traversal**: Walks outgoing edges from the target function to discover transitively needed helpers.
        3. **Budget Packing (MCKP)**: Selects between full function definitions and lightweight type stubs under a strict **2,048-token** budget.

        ### Pilot Empirical Validation (50 Tasks)
        | Approach | Pass Rate (35 Eligible) | Gain vs Baseline |
        |---|---|---|
        | **DA-RAG-B (Proposed)** | **8 / 35 (22.9%)** ✅ | **+100% (+4 Passes)** |
        | Dense Retrieval (UnixCoder) | 5 / 35 (14.3%) | +25% (+1 Pass) |
        | Dense 1-Hop Graph | 5 / 35 (14.3%) | +25% (+1 Pass) |
        | BM25 Lexical Retrieval | 4 / 35 (11.4%) | 0% |
        | **Baseline (None / Zero Context)** | **4 / 35 (11.4%)** | *Reference* |
        """)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  <div class="footer-text">
    🔬 <strong>DA-RAG Studio</strong> · Dependency-Aware Retrieval Augmented Generation · Qwen2.5-Coder-1.5B · DevEval Benchmark
  </div>
  <div class="footer-tag">Pilot: DA-RAG 8 vs None 4 Passes</div>
</div>
""", unsafe_allow_html=True)
