"""
DA-RAG Studio: Repository-Level Code Generation & Functional Testing Test Bench.
A single-page, live demonstration platform comparing:
- Baseline (none): Zero Repository Context
- DA-RAG-B (darag_b): AST Dependency Graph Retrieval & Budget Knapsack Packing
Powered by local Qwen2.5-Coder-1.5B-Instruct and Real Isolated Pytest Test Execution.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st
import pandas as pd

import repo_manager
import context_engine
import interactive_runner
import test_runner
import dag_visualizer

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DA-RAG Studio · Repository-Level Code Generation Test Bench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global Design System & CSS ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── DESIGN TOKENS ── */
:root {
    color-scheme: dark !important;
    --canvas: #07090E;
    --surface-1: #0D121F;
    --surface-2: #131A2B;
    --surface-3: #1A233A;

    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-mid: rgba(255, 255, 255, 0.15);
    --border-focus: #6366F1;

    --brand-500: #6366F1;
    --brand-400: #818CF8;
    --brand-glow: rgba(99, 102, 241, 0.25);

    --dag-500: #10B981;
    --dag-400: #34D399;
    --dag-bg: rgba(16, 185, 129, 0.1);
    --dag-border: rgba(16, 185, 129, 0.35);

    --none-500: #EF4444;
    --none-400: #F87171;
    --none-bg: rgba(239, 68, 68, 0.1);
    --none-border: rgba(239, 68, 68, 0.35);

    --warn-500: #F59E0B;
    --warn-400: #FBBF24;
    --warn-bg: rgba(245, 158, 11, 0.12);

    --text-pure: #FFFFFF;
    --text-primary: #F8FAFC;
    --text-secondary: #CBD5E1;
    --text-muted: #94A3B8;

    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}

html, body, .stApp {
    background-color: var(--canvas) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

.main .block-container {
    padding: 1.5rem 2.5rem 4rem !important;
    max-width: 1480px !important;
}

header[data-testid="stHeader"] {
    background: var(--canvas) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* ── STEP HEADER BAR ── */
.step-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 18px;
    margin: 28px 0 16px 0;
    background: linear-gradient(90deg, var(--surface-2) 0%, var(--surface-1) 100%);
    border-left: 4px solid var(--brand-500);
    border-radius: 6px;
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
}
.step-number {
    background: var(--brand-500);
    color: #FFFFFF;
    font-weight: 800;
    font-size: 0.8rem;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.05em;
}
.step-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-pure);
    letter-spacing: -0.01em;
}
.step-sub {
    font-size: 0.83rem;
    color: var(--text-muted);
    margin-left: auto;
}

/* ── CARDS & PANELS ── */
.panel-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.panel-card:hover {
    border-color: var(--border-mid);
}

/* ── STATUS PILLS & BADGES ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.status-pass {
    background: var(--dag-bg);
    color: var(--dag-400);
    border: 1px solid var(--dag-border);
}
.status-fail {
    background: var(--none-bg);
    color: var(--none-400);
    border: 1px solid var(--none-border);
}
.status-timeout {
    background: var(--warn-bg);
    color: var(--warn-400);
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.status-infra {
    background: rgba(148, 163, 184, 0.12);
    color: var(--text-muted);
    border: 1px solid rgba(148, 163, 184, 0.3);
}
.status-notrun {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
}

/* ── SIDE-BY-SIDE PANELS ── */
.method-card-none {
    background: var(--surface-1);
    border: 1px solid var(--none-border);
    border-radius: 10px;
    padding: 16px 20px;
}
.method-card-dag {
    background: var(--surface-1);
    border: 1px solid var(--dag-border);
    border-radius: 10px;
    padding: 16px 20px;
}
.method-header-none {
    color: var(--none-400);
    font-size: 0.95rem;
    font-weight: 700;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.method-header-dag {
    color: var(--dag-400);
    font-size: 0.95rem;
    font-weight: 700;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

/* ── SUMMARY STAT CALLOUT ── */
.summary-deck {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 16px 0;
}
.summary-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}
.summary-card-val {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-pure);
    font-family: var(--font-mono);
}
.summary-card-lbl {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Comparison Table Styling */
.table-metric-head {
    font-weight: 700;
    color: var(--text-pure);
    background: var(--surface-2);
}
</style>
""", unsafe_allow_html=True)

# ─── Top App Header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-subtle); padding-bottom:14px; margin-bottom:16px;">
  <div>
    <h2 style="margin:0; font-size:1.6rem; font-weight:800; color:var(--text-pure); letter-spacing:-0.02em;">
      ⚡ DA-RAG Studio <span style="font-size:0.95rem; font-weight:500; color:var(--brand-400); margin-left:8px;">Live Code Generation & Functional Test Bench</span>
    </h2>
    <div style="font-size:0.85rem; color:var(--text-muted); margin-top:4px;">
      Empirical Repository-Level Code Generation: Baseline (Zero Retrieval) vs DA-RAG-B (AST Dependency Graph Retrieval & Knapsack Packing)
    </div>
  </div>
  <div style="display:flex; gap:10px; align-items:center;">
    <span class="status-pill status-pass">Qwen2.5-Coder-1.5B</span>
    <span class="status-pill status-pass">NVIDIA RTX 4050</span>
    <span class="status-pill status-pass">2,048 Tokens</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Session State Initialization ────────────────────────────────────────────
if "repo_path" not in st.session_state:
    # Default to standard DevEval exemplar repo
    st.session_state.repo_path = "/home/Suriya/darag_data/Source_Code/System/fs"
if "repo_meta" not in st.session_state:
    st.session_state.repo_meta = None
if "target_file" not in st.session_state:
    st.session_state.target_file = "fs/wildcard.py"
if "func_name" not in st.session_state:
    st.session_state.func_name = "match_any"
if "target_sig" not in st.session_state:
    st.session_state.target_sig = "def match_any(patterns, name):"
if "target_doc" not in st.session_state:
    st.session_state.target_doc = "Check if a path matches any pattern in a wildcard list. If patterns is empty, return True."
if "test_cmd" not in st.session_state:
    st.session_state.test_cmd = "pytest tests/test_wildcard.py -v"
if "context_result" not in st.session_state:
    st.session_state.context_result = None
if "gen_results" not in st.session_state:
    st.session_state.gen_results = None
if "test_results" not in st.session_state:
    st.session_state.test_results = None
if "user_approved_tests" not in st.session_state:
    st.session_state.user_approved_tests = True

# Pre-load repository metadata if not yet loaded
if st.session_state.repo_meta is None and st.session_state.repo_path:
    st.session_state.repo_meta = repo_manager.inspect_repository(st.session_state.repo_path)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: SELECT REPOSITORY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-banner">
  <span class="step-number">STEP 1</span>
  <span class="step-title">Select Repository</span>
  <span class="step-sub">Select a local Python repository on your computer</span>
</div>
""", unsafe_allow_html=True)

# Quick repository selectors for live demonstration convenience
quick_col1, quick_col2, quick_col3, quick_col4 = st.columns([1, 1, 1, 1.2])
with quick_col1:
    if st.button("📁 System / fs (Wildcard)", use_container_width=True, help="Standard filesystem repo with test suite"):
        st.session_state.repo_path = "/home/Suriya/darag_data/Source_Code/System/fs"
        st.session_state.repo_meta = repo_manager.inspect_repository(st.session_state.repo_path)
        st.session_state.target_file = "fs/wildcard.py"
        st.session_state.func_name = "match_any"
        st.session_state.target_sig = "def match_any(patterns, name):"
        st.session_state.target_doc = "Check if a path matches any pattern in a wildcard list. If patterns is empty, return True."
        st.session_state.test_cmd = "pytest tests/test_wildcard.py -v"
        st.session_state.context_result = None
        st.session_state.gen_results = None
        st.session_state.test_results = None
        st.rerun()

with quick_col2:
    if st.button("📁 Twtxt (Config)", use_container_width=True, help="Twtxt feed repository"):
        st.session_state.repo_path = "/home/Suriya/darag_data/Source_Code/Communications/twtxt"
        st.session_state.repo_meta = repo_manager.inspect_repository(st.session_state.repo_path)
        st.session_state.target_file = "twtxt/config.py"
        st.session_state.func_name = "get_following"
        st.session_state.target_sig = "def get_following(self):"
        st.session_state.target_doc = "Return list of followed users from the parsed twtxt config."
        st.session_state.test_cmd = "pytest tests/test_config.py -v"
        st.session_state.context_result = None
        st.session_state.gen_results = None
        st.session_state.test_results = None
        st.rerun()

with quick_col3:
    if st.button("📁 DA-RAG Engine", use_container_width=True, help="Current DA-RAG project repository"):
        st.session_state.repo_path = "/home/Suriya/projects/da-rag"
        st.session_state.repo_meta = repo_manager.inspect_repository(st.session_state.repo_path)
        st.session_state.target_file = "src/darag/analysis.py"
        st.session_state.func_name = "format_summary"
        st.session_state.target_sig = "def format_summary(results: list) -> str:"
        st.session_state.target_doc = "Format evaluation run results into a human-readable summary string."
        st.session_state.test_cmd = "pytest tests/test_cli_smoke.py -v"
        st.session_state.context_result = None
        st.session_state.gen_results = None
        st.session_state.test_results = None
        st.rerun()

with quick_col4:
    st.caption("💡 Supports POSIX paths (`/home/...`) and Windows drive paths (`C:\\...`, `/mnt/c/...`).")

path_col, btn_col = st.columns([4, 1])
with path_col:
    input_path = st.text_input(
        "Repository Folder Path",
        value=st.session_state.repo_path,
        placeholder="/home/Suriya/projects/my-repo or C:\\Users\\... or /mnt/c/...",
        label_visibility="collapsed"
    )
with btn_col:
    load_clicked = st.button("📥 Load Repository", type="primary", use_container_width=True)

if load_clicked:
    st.session_state.repo_path = input_path.strip()
    st.session_state.repo_meta = repo_manager.inspect_repository(input_path)
    st.session_state.context_result = None
    st.session_state.gen_results = None
    st.session_state.test_results = None
    if st.session_state.repo_meta and not st.session_state.repo_meta.error:
        if st.session_state.repo_meta.root_path:
            st.session_state.repo_path = st.session_state.repo_meta.root_path
        if st.session_state.repo_meta.source_files:
            st.session_state.target_file = st.session_state.repo_meta.source_files[0]
            st.session_state.test_cmd = st.session_state.repo_meta.proposed_test_cmd
    st.rerun()

repo_meta: Optional[repo_manager.RepoMetadata] = st.session_state.repo_meta

if repo_meta:
    if repo_meta.error:
        st.error(f"❌ {repo_meta.error}")
    else:
        # Repository telemetry card
        kf_badges = " ".join([f"`{k}`" for k in repo_meta.key_files.keys()]) if repo_meta.key_files else "None"
        st.markdown(f"""
        <div class="panel-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div>
              <span style="font-size:1.15rem; font-weight:700; color:var(--text-pure);">📦 {repo_meta.project_name}</span>
              <span style="font-size:0.85rem; color:var(--text-muted); margin-left:10px;"><code>{repo_meta.root_path}</code></span>
            </div>
            <div>
              <span class="status-pill status-pass">Language: {repo_meta.language}</span>
              <span class="status-pill status-pass">{len(repo_meta.source_files)} Source Files</span>
              <span class="status-pill status-pass">{len(repo_meta.test_files)} Test Files</span>
            </div>
          </div>
          <div style="display:flex; gap:20px; font-size:0.83rem; color:var(--text-secondary); flex-wrap:wrap;">
            <span><strong>Framework:</strong> <code>{repo_meta.test_framework}</code></span>
            <span><strong>Discovered Key Files:</strong> {kf_badges}</span>
            <span><strong>Security Boundary:</strong> <span style="color:var(--dag-400);">Active (Secrets, .git, venvs filtered)</span></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"📂 View Repository File Tree Preview ({len(repo_meta.source_files) + len(repo_meta.test_files)} files total)", expanded=False):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Source Files:**")
                st.code("\n".join(repo_meta.source_files[:25]) or "No source files", language="text")
            with t_col2:
                st.markdown("**Test Files:**")
                st.code("\n".join(repo_meta.test_files[:25]) or "No test files", language="text")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: DEFINE FUNCTION TO IMPLEMENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-banner">
  <span class="step-number">STEP 2</span>
  <span class="step-title">Define Function to Implement</span>
  <span class="step-sub">Select a target file, pick an existing stub or enter signature and requirements</span>
</div>
""", unsafe_allow_html=True)

if repo_meta and not repo_meta.error and repo_meta.source_files:
    f_col1, f_col2 = st.columns([1.5, 2.5], gap="large")

    with f_col1:
        st.markdown('<p style="font-size:0.85rem; font-weight:600; color:var(--text-secondary); margin-bottom:6px;">Target Python File in Repository:</p>', unsafe_allow_html=True)
        all_sources = repo_meta.source_files
        curr_idx = all_sources.index(st.session_state.target_file) if st.session_state.target_file in all_sources else 0
        selected_file = st.selectbox(
            "Target File",
            all_sources,
            index=curr_idx,
            label_visibility="collapsed"
        )
        if selected_file != st.session_state.target_file:
            st.session_state.target_file = selected_file
            # Propose matching test file if available
            matching_test = repo_manager.find_matching_test_file(repo_meta.root_path, selected_file)
            if matching_test:
                st.session_state.test_cmd = f"pytest {matching_test} -v"
            st.rerun()

        # Extract functions/stubs from selected file using AST
        stubs = repo_manager.extract_functions_from_file(repo_meta.root_path, st.session_state.target_file)
        if stubs:
            stub_options = ["-- Custom Entry --"] + [f"{'⚡ [STUB] ' if s.is_stub else ''}{s.qualname} (line {s.start_line})" for s in stubs]
            chosen_stub = st.selectbox(
                "Prefill from Function / Stub in File:",
                stub_options,
                index=0,
                help="Select an existing function or unimplemented stub from the target file."
            )
            if chosen_stub != "-- Custom Entry --":
                clean_name = chosen_stub.replace("⚡ [STUB] ", "").split(" (line ")[0].strip()
                s_obj = next((s for s in stubs if s.qualname == clean_name), None)
                if s_obj:
                    st.session_state.func_name = s_obj.name
                    st.session_state.target_sig = s_obj.signature
                    st.session_state.target_doc = s_obj.docstring or f"Implement the {s_obj.name} function according to repository contracts."
        else:
            st.info("No top-level functions found in this file. Enter custom signature below.")

    with f_col2:
        fn_name_input = st.text_input(
            "Function Name",
            value=st.session_state.func_name,
            placeholder="e.g. match_any"
        )
        st.session_state.func_name = fn_name_input.strip()

        sig_input = st.text_input(
            "Function Signature / Header",
            value=st.session_state.target_sig,
            placeholder="def match_any(patterns, name):"
        )
        st.session_state.target_sig = sig_input.strip()

        doc_input = st.text_area(
            "Requirements / Docstring",
            value=st.session_state.target_doc,
            height=90,
            placeholder="Docstring, specifications, edge cases, or parameter constraints..."
        )
        st.session_state.target_doc = doc_input.strip()

        # Combine into canonical prompt
        combined_prompt_code = f"{st.session_state.target_sig}\n    \"\"\"\n    {st.session_state.target_doc}\n    \"\"\"\n"

        cmd_col1, cmd_col2 = st.columns([3, 1])
        with cmd_col1:
            test_cmd_input = st.text_input(
                "Test Command (Executed against isolated temporary repository)",
                value=st.session_state.test_cmd,
                placeholder="pytest tests/test_wildcard.py -v"
            )
            st.session_state.test_cmd = test_cmd_input.strip()
            if "<" in st.session_state.test_cmd and ">" in st.session_state.test_cmd:
                st.caption("💡 **Tip:** Replace any placeholder like `<matching_test_file>` with a concrete test file (e.g. `pytest tests/test_admin_session_management.py -v`). If left as `<...>`, DA-RAG will automatically substitute the matching test file.")
        with cmd_col2:
            st.markdown('<p style="font-size:0.75rem; color:var(--text-muted); margin-top:28px;">Target: Isolated Copy</p>', unsafe_allow_html=True)
else:
    st.warning("Please enter and load a valid repository path above to proceed.")
    combined_prompt_code = ""


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: RETRIEVE REPOSITORY CONTEXT (DA-RAG-B)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-banner">
  <span class="step-number">STEP 3</span>
  <span class="step-title">Retrieve Repository Context</span>
  <span class="step-sub">Inspect genuine DA-RAG-B AST dependency retrieval and budget knapsack packing</span>
</div>
""", unsafe_allow_html=True)

retr_col1, retr_col2 = st.columns([1, 4])
with retr_col1:
    retrieve_btn = st.button("🔍 Retrieve Context (DA-RAG-B)", type="secondary", use_container_width=True)

if retrieve_btn or (st.session_state.context_result is None and repo_meta and not repo_meta.error):
    with st.spinner("Extracting AST dependency graph and executing DA-RAG-B multi-hop retrieval..."):
        ctx_res = context_engine.retrieve_darag_context(
            repo_root=st.session_state.repo_path,
            target_file_rel=st.session_state.target_file,
            func_name=st.session_state.func_name,
            docstring_or_spec=st.session_state.target_doc,
            target_signature=st.session_state.target_sig,
            budget_tokens=2048,
            k=10,
            hops=2,
            lambda_weight=0.5
        )
        st.session_state.context_result = ctx_res

ctx_res: Optional[context_engine.RetrievalResult] = st.session_state.context_result

if ctx_res:
    if ctx_res.error:
        st.error(f"❌ Retrieval Error: {ctx_res.error}")
    else:
        # Comparison between Method A and Method B contexts
        c_col_none, c_col_dag = st.columns(2, gap="medium")

        with c_col_none:
            st.markdown("""
            <div class="method-card-none">
              <div class="method-header-none">
                <span>🚫 Method A · No Retrieval (none)</span>
                <span class="status-pill status-fail">0 / 2,048 Tokens (0%)</span>
              </div>
              <div style="background:rgba(0,0,0,0.25); border:1px solid var(--border-subtle); border-radius:6px; padding:28px 16px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:6px;">⬛</div>
                <div style="font-weight:700; color:var(--text-pure); font-size:0.95rem;">Zero Repository Context Provided</div>
                <div style="font-size:0.8rem; color:var(--text-muted); margin-top:4px;">
                  Model receives only the isolated function signature and docstring.
                  Zero helper functions, class contracts, or internal sentinel values are injected.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c_col_dag:
            pct_used = int((ctx_res.context_tokens / ctx_res.token_budget) * 100) if ctx_res.token_budget else 0
            st.markdown(f"""
            <div class="method-card-dag">
              <div class="method-header-dag">
                <span>🌿 Method B · DA-RAG-B (darag_b)</span>
                <span class="status-pill status-pass">{ctx_res.context_tokens} / {ctx_res.token_budget} Tokens ({pct_used}%) · {len(ctx_res.packed_items)} Units</span>
              </div>
              <div style="font-size:0.8rem; color:var(--text-secondary); margin-bottom:8px;">
                <strong>Retrieved Files:</strong> {", ".join([f'<code>{f}</code>' for f in ctx_res.retrieved_files]) or 'None'}<br>
                <strong>Key Symbols:</strong> {", ".join([f'<code>{s}</code>' for s in ctx_res.retrieved_symbols[:6]]) or 'None'}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Interactive SVG Dependency Graph
            svg_html = dag_visualizer.render_dag_graph_svg(
                target_name=st.session_state.func_name,
                packed_items=ctx_res.packed_items,
                tokens=ctx_res.context_tokens,
                n_units=len(ctx_res.packed_items)
            )
            st.markdown(svg_html, unsafe_allow_html=True)

        # Context inspection expanders
        exp_col1, exp_col2 = st.columns(2, gap="medium")
        with exp_col1:
            with st.expander(f"📦 View Packed Repository Code Supplied to Qwen ({ctx_res.context_tokens} tokens)", expanded=False):
                st.code(ctx_res.packed_context_text or "# No context units packed", language="python")
        with exp_col2:
            with st.expander(f"📑 View Packed Units Details & Scores ({len(ctx_res.packed_items)} units)", expanded=False):
                if ctx_res.packed_items:
                    st.dataframe(pd.DataFrame(ctx_res.packed_items)[["symbol", "file", "kind", "mode", "cost", "score", "relation"]], use_container_width=True)
                if ctx_res.omitted_items:
                    st.markdown("**Omitted Candidates Due to Token Budget:**")
                    st.dataframe(pd.DataFrame(ctx_res.omitted_items)[["rank", "symbol", "file", "kind", "estimated_cost", "reason"]], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE BOTH IMPLEMENTATIONS USING QWEN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-banner">
  <span class="step-number">STEP 4</span>
  <span class="step-title">Generate Both Implementations Using Qwen</span>
  <span class="step-sub">Symmetric inference with local Qwen2.5-Coder-1.5B-Instruct on GPU</span>
</div>
""", unsafe_allow_html=True)

gen_btn = st.button("🚀  Generate Both Implementations (Qwen2.5-Coder)", type="primary", use_container_width=True)

if gen_btn:
    if not st.session_state.func_name or not st.session_state.target_sig:
        st.error("Please specify a valid function name and signature in Step 2.")
    else:
        with st.spinner("Generating implementations for both Method A and Method B using Qwen2.5-Coder..."):
            darag_ctx = ctx_res.packed_context_text if ctx_res else ""
            input_prompt = f"{st.session_state.target_sig}\n    \"\"\"\n    {st.session_state.target_doc}\n    \"\"\"\n"
            
            gen_res = interactive_runner.generate_dual_implementations(
                func_name=st.session_state.func_name,
                input_code=input_prompt,
                darag_context_text=darag_ctx,
                max_new_tokens=384
            )
            # Add context units metadata
            gen_res["none"]["context_units"] = 0
            gen_res["none"]["context_tokens"] = 0
            gen_res["darag_b"]["context_units"] = len(ctx_res.packed_items) if ctx_res else 0
            gen_res["darag_b"]["context_tokens"] = ctx_res.context_tokens if ctx_res else 0

            st.session_state.gen_results = gen_res
            st.session_state.test_results = None  # Reset tests when fresh code is generated

gen_results = st.session_state.gen_results

if gen_results:
    res_none = gen_results["none"]
    res_dag = gen_results["darag_b"]

    g_col1, g_col2 = st.columns(2, gap="medium")

    with g_col1:
        st.markdown(f"""
        <div class="method-card-none">
          <div class="method-header-none">
            <span>Method A: No Retrieval (none)</span>
            <span class="status-pill status-fail">Tokens: {res_none['tokens']} · {res_none['latency']:.2f}s</span>
          </div>
          <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:8px;">
            Syntax: <strong>{res_none['syntax_msg']}</strong> | Context Injected: <strong>0 tokens</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.code(res_none["completion"] or "# Empty completion", language="python")
        with st.expander("🔍 View Raw Baseline Prompt Sent to Qwen", expanded=False):
            st.code(res_none["prompt_text"], language="text")

    with g_col2:
        st.markdown(f"""
        <div class="method-card-dag">
          <div class="method-header-dag">
            <span>Method B: DA-RAG-B (darag_b)</span>
            <span class="status-pill status-pass">Tokens: {res_dag['tokens']} · {res_dag['latency']:.2f}s</span>
          </div>
          <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:8px;">
            Syntax: <strong>{res_dag['syntax_msg']}</strong> | Context Injected: <strong>{res_dag['context_tokens']} tokens</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.code(res_dag["completion"] or "# Empty completion", language="python")
        with st.expander("🔍 View Raw DA-RAG Prompt Sent to Qwen", expanded=False):
            st.code(res_dag["prompt_text"], language="text")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: RUN REAL TESTS FOR BOTH METHODS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-banner">
  <span class="step-number">STEP 5</span>
  <span class="step-title">Run Real Tests for Both Methods</span>
  <span class="step-sub">Execute actual repository test suite in isolated temporary worktrees</span>
</div>
""", unsafe_allow_html=True)

if gen_results:
    t_ctrl1, t_ctrl2 = st.columns([3, 1])
    with t_ctrl1:
        st.session_state.user_approved_tests = st.checkbox(
            "🛡️ Explicit Approval: Run test command in isolated disposable sandbox copies of this repository",
            value=st.session_state.user_approved_tests,
            help="Creates separate temporary copies in /tmp, patches the target file, executes pytest, and cleans up."
        )
    with t_ctrl2:
        run_tests_btn = st.button("🧪  Run Tests on Both Implementations", type="primary", use_container_width=True)

    if run_tests_btn:
        if not st.session_state.user_approved_tests:
            st.warning("Please check the explicit approval box before executing tests.")
        elif not st.session_state.test_cmd:
            st.error("Please specify a valid test command in Step 2.")
        else:
            with st.spinner("Running tests in isolated temporary repository copies..."):
                test_none = test_runner.run_test_in_isolated_copy(
                    repo_root=st.session_state.repo_path,
                    target_file_rel=st.session_state.target_file,
                    func_name=st.session_state.func_name,
                    generated_code=res_none["completion"],
                    test_command=st.session_state.test_cmd,
                    method="none",
                    timeout_s=30.0,
                    target_signature=st.session_state.target_sig
                )

                test_dag = test_runner.run_test_in_isolated_copy(
                    repo_root=st.session_state.repo_path,
                    target_file_rel=st.session_state.target_file,
                    func_name=st.session_state.func_name,
                    generated_code=res_dag["completion"],
                    test_command=st.session_state.test_cmd,
                    method="darag_b",
                    timeout_s=30.0,
                    target_signature=st.session_state.target_sig
                )

                st.session_state.test_results = {
                    "none": test_none,
                    "darag_b": test_dag
                }
else:
    st.info("Generate implementations in Step 4 above before running tests.")

test_results = st.session_state.test_results

if test_results:
    t_none: test_runner.TestExecutionResult = test_results["none"]
    t_dag: test_runner.TestExecutionResult = test_results["darag_b"]

    def status_badge(status: str) -> str:
        if status == "PASS":
            return '<span class="status-pill status-pass">✓ PASS</span>'
        elif status == "FAIL":
            return '<span class="status-pill status-fail">✗ FAIL</span>'
        elif status == "TIMEOUT":
            return '<span class="status-pill status-timeout">⏱ TIMEOUT</span>'
        elif status == "INFRASTRUCTURE ERROR":
            return '<span class="status-pill status-infra">⚙ INFRASTRUCTURE ERROR</span>'
        elif status == "GENERATION ERROR":
            return '<span class="status-pill status-fail">⚠️ GENERATION ERROR</span>'
        else:
            return '<span class="status-pill status-notrun">NOT RUN</span>'

    t_res_col1, t_res_col2 = st.columns(2, gap="medium")

    with t_res_col1:
        st.markdown(f"""
        <div class="method-card-none">
          <div class="method-header-none">
            <span>Method A Test Outcome: {status_badge(t_none.status)}</span>
            <span style="font-size:0.8rem; font-family:var(--font-mono); color:var(--text-muted);">Duration: {t_none.duration_s:.2f}s</span>
          </div>
          <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:8px;">
            Passed: <strong style="color:var(--dag-400);">{t_none.tests_passed}</strong> |
            Failed: <strong style="color:var(--none-400);">{t_none.tests_failed}</strong> |
            Exit Code: <code>{t_none.exit_code}</code>
          </div>
          {f'<div style="font-size:0.8rem; color:var(--none-400);">{t_none.error_details}</div>' if t_none.error_details else ''}
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📄 View Full Test Stdout / Stderr (Method A)", expanded=False):
            st.code(t_none.stdout or "(No stdout captured)", language="text")
            if t_none.stderr:
                st.code(t_none.stderr, language="text")

    with t_res_col2:
        st.markdown(f"""
        <div class="method-card-dag">
          <div class="method-header-dag">
            <span>Method B Test Outcome: {status_badge(t_dag.status)}</span>
            <span style="font-size:0.8rem; font-family:var(--font-mono); color:var(--text-muted);">Duration: {t_dag.duration_s:.2f}s</span>
          </div>
          <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:8px;">
            Passed: <strong style="color:var(--dag-400);">{t_dag.tests_passed}</strong> |
            Failed: <strong style="color:var(--none-400);">{t_dag.tests_failed}</strong> |
            Exit Code: <code>{t_dag.exit_code}</code>
          </div>
          {f'<div style="font-size:0.8rem; color:var(--none-400);">{t_dag.error_details}</div>' if t_dag.error_details else ''}
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📄 View Full Test Stdout / Stderr (Method B)", expanded=False):
            st.code(t_dag.stdout or "(No stdout captured)", language="text")
            if t_dag.stderr:
                st.code(t_dag.stderr, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: RESULTS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-banner">
  <span class="step-number">STEP 6</span>
  <span class="step-title">Results Comparison</span>
  <span class="step-sub">Side-by-side empirical scorecard based strictly on executed tests</span>
</div>
""", unsafe_allow_html=True)

if test_results and gen_results and ctx_res:
    t_none: test_runner.TestExecutionResult = test_results["none"]
    t_dag: test_runner.TestExecutionResult = test_results["darag_b"]

    # Factual result summary
    if t_dag.status == "PASS" and t_none.status != "PASS":
        verdict_text = f"🏆 <strong>DA-RAG-B passed {t_dag.tests_passed} tests; no-retrieval passed {t_none.tests_passed} tests.</strong>"
        v_bg = "var(--dag-bg)"
        v_border = "var(--dag-border)"
        narrative = (
            f"The baseline (none) failed due to missing repository-internal helpers. "
            f"DA-RAG-B retrieved relevant definitions ({', '.join(ctx_res.retrieved_symbols[:3]) or 'internal helpers'}), "
            f"enabling Qwen to satisfy all functional assertions in <code>{st.session_state.test_cmd}</code>."
        )
    elif t_none.status == "PASS" and t_dag.status == "PASS":
        verdict_text = f"🤝 <strong>Both implementations passed functional tests ({t_dag.tests_passed} tests).</strong>"
        v_bg = "rgba(99, 102, 241, 0.12)"
        v_border = "rgba(99, 102, 241, 0.35)"
        narrative = "The task specification was self-contained enough that both implementations satisfied the test suite."
    elif t_dag.status == "FAIL" and t_none.status == "FAIL":
        verdict_text = f"⚠️ <strong>Both implementations failed test assertions (Passed: DA-RAG-B {t_dag.tests_passed}, None {t_none.tests_passed}).</strong>"
        v_bg = "rgba(245, 158, 11, 0.1)"
        v_border = "rgba(245, 158, 11, 0.3)"
        narrative = "Neither implementation satisfied all assertions. Review test logs above for specific failure traces."
    else:
        verdict_text = f"📊 <strong>Test Results: DA-RAG-B ({t_dag.status}) vs None ({t_none.status})</strong>"
        v_bg = "var(--surface-2)"
        v_border = "var(--border-subtle)"
        narrative = f"Method A: {t_none.status} ({t_none.error_details or 'exited ' + str(t_none.exit_code)}). Method B: {t_dag.status} ({t_dag.error_details or 'exited ' + str(t_dag.exit_code)})."

    st.markdown(f"""
    <div style="background:{v_bg}; border:1px solid {v_border}; border-radius:8px; padding:16px 20px; margin-bottom:18px;">
      <div style="font-size:1.1rem; color:var(--text-pure); margin-bottom:6px;">{verdict_text}</div>
      <div style="font-size:0.86rem; color:var(--text-secondary); line-height:1.5;">{narrative}</div>
    </div>
    """, unsafe_allow_html=True)

    # Professional comparison table matching Section 8 requirements
    comparison_data = [
        {"Metric": "Repository context units", "No Retrieval (none)": "0", "DA-RAG-B (darag_b)": str(len(ctx_res.packed_items))},
        {"Metric": "Context tokens", "No Retrieval (none)": "0", "DA-RAG-B (darag_b)": f"{ctx_res.context_tokens} / {ctx_res.token_budget}"},
        {"Metric": "Generation latency", "No Retrieval (none)": f"{res_none['latency']:.2f}s", "DA-RAG-B (darag_b)": f"{res_dag['latency']:.2f}s"},
        {"Metric": "Syntax validation", "No Retrieval (none)": res_none['syntax_msg'], "DA-RAG-B (darag_b)": res_dag['syntax_msg']},
        {"Metric": "Tests passed", "No Retrieval (none)": str(t_none.tests_passed), "DA-RAG-B (darag_b)": str(t_dag.tests_passed)},
        {"Metric": "Tests failed", "No Retrieval (none)": str(t_none.tests_failed), "DA-RAG-B (darag_b)": str(t_dag.tests_failed)},
        {"Metric": "Test execution status", "No Retrieval (none)": t_none.status, "DA-RAG-B (darag_b)": t_dag.status},
        {"Metric": "Error details", "No Retrieval (none)": t_none.error_details or "None", "DA-RAG-B (darag_b)": t_dag.error_details or "None"},
    ]

    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp.set_index("Metric"))

    # Side-by-side code review
    st.markdown('<p style="font-size:0.95rem; font-weight:700; color:var(--text-pure); margin-top:20px; margin-bottom:8px;">Code Implementation Comparison:</p>', unsafe_allow_html=True)
    c_view1, c_view2 = st.columns(2, gap="medium")
    with c_view1:
        st.markdown(f"**Baseline (`none`) Completion** — Status: `{t_none.status}`")
        st.code(res_none["completion"], language="python")
    with c_view2:
        st.markdown(f"**DA-RAG-B (`darag_b`) Completion** — Status: `{t_dag.status}`")
        st.code(res_dag["completion"], language="python")
else:
    st.info("Execute Step 4 (Generation) and Step 5 (Tests) to render the full empirical scorecard.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px; padding-top:16px; border-top:1px solid var(--border-subtle); display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:var(--text-muted);">
  <div>🔬 <strong>DA-RAG Studio</strong> · Repository-Level Code Generation Test Bench · Qwen2.5-Coder-1.5B · Isolated Pytest Runner</div>
  <div>WSL 2 Ubuntu · CUDA Enabled</div>
</div>
""", unsafe_allow_html=True)
