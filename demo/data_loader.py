"""
Data access layer for DA-RAG Interactive Code Generation Test Bench.
Provides strict READ-ONLY access to real experiment records and artifacts.
Gracefully handles concurrent writes to in-progress experiment JSONL files.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import sys

# Dynamic path resolution supporting both Linux/WSL and Windows hosts
if sys.platform == "win32":
    DATA_DIR = Path(r"C:\Users\suriy\darag_data")
    _wsl_proj = Path(r"\\wsl$\Ubuntu\home\Suriya\projects\da-rag")
    _local_proj = Path(r"C:\Users\suriy\OneDrive\Desktop\VS CODE\NLP\da-rag")
    PROJECT_ROOT = _wsl_proj if _wsl_proj.exists() else _local_proj
else:
    PROJECT_ROOT = Path('/home/Suriya/projects/da-rag')
    DATA_DIR = Path('/home/Suriya/darag_data')

META_FILE = DATA_DIR / 'meta' / 'data.jsonl'
DEV_EVAL_PROMPTS = DATA_DIR / 'DevEval_repo' / 'Experiments' / 'prompt' / 'LM_prompt_elements.jsonl'
ELIGIBLE_TASKS_FILE = PROJECT_ROOT / 'results' / 'eligible_tasks.json'
INELIGIBLE_TASKS_FILE = PROJECT_ROOT / 'results' / 'ineligible_tasks.json'
MAPPING_ERROR_TASKS_FILE = PROJECT_ROOT / 'results' / 'mapping_error_tasks.json'
PILOT_RESULTS_CSV = PROJECT_ROOT / 'results' / 'per_task_results_pilot.csv'

PILOT_GEN_DIR = DATA_DIR / 'runs' / 'generation' / 'gen-c812ca3ce189f599-pilot'
FULL_GEN_DIR = DATA_DIR / 'runs' / 'generation' / 'gen-c812ca3ce189f599-full'
RETRIEVAL_FILE = DATA_DIR / 'runs' / 'retrieval' / 'retr-e7281fcc0a7c9a40-full' / 'darag.jsonl'
SOURCE_CODE_DIR = DATA_DIR / 'Source_Code'



def read_jsonl_safely(path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Reads a JSONL file in strict read-only mode.
    If the file is actively being appended to, an incomplete trailing line
    is ignored and had_incomplete_line is returned as True.
    Never attempts to repair or modify the file on disk.
    """
    if not path.exists():
        return [], False

    records = []
    had_incomplete_line = False

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            if idx == len(lines) - 1:
                # Incomplete trailing line from active concurrent append
                had_incomplete_line = True
            else:
                # Malformed non-terminal line, skip safely without modifying file
                pass

    return records, had_incomplete_line


def load_eligibility_sets() -> Dict[str, Any]:
    """Loads the ground-truth eligibility manifests and sets."""
    eligible = set()
    ineligible = set()
    mapping_error = set()

    if ELIGIBLE_TASKS_FILE.exists():
        try:
            with open(ELIGIBLE_TASKS_FILE, 'r', encoding='utf-8') as f:
                eligible = set(json.load(f))
        except Exception:
            pass

    if INELIGIBLE_TASKS_FILE.exists():
        try:
            with open(INELIGIBLE_TASKS_FILE, 'r', encoding='utf-8') as f:
                ineligible = set(json.load(f))
        except Exception:
            pass

    if MAPPING_ERROR_TASKS_FILE.exists():
        try:
            with open(MAPPING_ERROR_TASKS_FILE, 'r', encoding='utf-8') as f:
                mapping_error = set(json.load(f))
        except Exception:
            pass

    return {
        'eligible': eligible,
        'ineligible': ineligible,
        'mapping_error': mapping_error
    }


def load_all_task_metadata() -> Dict[str, Dict[str, Any]]:
    """Loads task metadata from meta/data.jsonl and LM_prompt_elements.jsonl."""
    tasks = {}
    if META_FILE.exists():
        recs, _ = read_jsonl_safely(META_FILE)
        for r in recs:
            tid = r.get('namespace')
            if tid:
                tasks[tid] = r

    # Merge in prompt elements (input_code, class_name, function_name)
    if DEV_EVAL_PROMPTS.exists():
        elems, _ = read_jsonl_safely(DEV_EVAL_PROMPTS)
        for el in elems:
            tid = el.get('namespace')
            if tid in tasks:
                tasks[tid]['input_code'] = el.get('input_code', '')
                tasks[tid]['prompt_elements'] = el
            elif tid:
                tasks[tid] = el
                tasks[tid]['input_code'] = el.get('input_code', '')

    return tasks


def load_pilot_summary_csv() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Loads results from per_task_results_pilot.csv keyed by (task_id, method)."""
    out = {}
    if PILOT_RESULTS_CSV.exists():
        with open(PILOT_RESULTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get('task_id')
                method = row.get('method')
                if tid and method:
                    out[(tid, method)] = row
    return out


def load_run_predictions(run_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], bool]:
    """
    Loads predictions.jsonl from the given run directory.
    Returns (dict_keyed_by_(task_id, method), had_incomplete_line).
    """
    pred_path = run_dir / 'predictions.jsonl'
    recs, incomplete = read_jsonl_safely(pred_path)
    out = {}
    for r in recs:
        tid = r.get('task_id')
        method = r.get('method')
        if tid and method:
            out[(tid, method)] = r
    return out, incomplete


def load_run_prompts(run_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], bool]:
    """
    Loads prompts.jsonl from the given run directory.
    Returns (dict_keyed_by_(task_id, method), had_incomplete_line).
    """
    prompts_path = run_dir / 'prompts.jsonl'
    recs, incomplete = read_jsonl_safely(prompts_path)
    out = {}
    for r in recs:
        tid = r.get('task_id')
        method = r.get('method')
        if tid and method:
            out[(tid, method)] = r
    return out, incomplete


def load_run_functional_eval(run_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], bool]:
    """
    Loads functional_eval.jsonl from the given run directory.
    Returns (dict_keyed_by_(task_id, method), had_incomplete_line).
    """
    eval_path = run_dir / 'functional_eval.jsonl'
    if not eval_path.exists():
        return {}, False
    recs, incomplete = read_jsonl_safely(eval_path)
    out = {}
    for r in recs:
        tid = r.get('task_id')
        method = r.get('method')
        if tid and method:
            out[(tid, method)] = r
    return out, incomplete


def load_retrieval_candidates_for_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Scans the frozen darag.jsonl retrieval file for the specified task_id.
    Returns the candidate record dict or None.
    """
    if not RETRIEVAL_FILE.exists():
        return None

    needle = f'"task_id": "{task_id}"'
    try:
        with open(RETRIEVAL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if needle in line:
                    d = json.loads(line)
                    if d.get('task_id') == task_id:
                        return d
    except Exception:
        pass
    return None


def extract_context_and_prompt(prompt_text: str) -> Tuple[str, str]:
    """
    Extracts the retrieved code context and the user prompt from a full chat prompt string.
    """
    if not prompt_text:
        return "", ""

    ctx = ""
    user_prompt = prompt_text

    marker_intro = 'The following code from the same repository may be relevant (read-only context):'
    marker_start = marker_intro + '\n```python\n'
    marker_end = '\n```\n'

    if marker_intro in prompt_text:
        idx1 = prompt_text.find(marker_start)
        if idx1 != -1:
            s_idx = idx1 + len(marker_start)
            idx2 = prompt_text.find(marker_end, s_idx)
            if idx2 != -1:
                ctx = prompt_text[s_idx:idx2].strip()
                user_prompt = prompt_text[idx2 + len(marker_end):].strip()
            else:
                ctx = prompt_text[s_idx:].strip()
                user_prompt = ""
    elif '<|im_start|>user\n' in prompt_text:
        parts = prompt_text.split('<|im_start|>user\n', 1)
        if len(parts) > 1:
            user_prompt = parts[1]

    # Clean chat end tags
    if '<|im_end|>' in user_prompt:
        user_prompt = user_prompt.split('<|im_end|>')[0].strip()

    return ctx, user_prompt


def get_functional_outcome_label(eval_rec: Optional[Dict[str, Any]], pred_rec: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    """
    Evaluates the official functional test outcome strictly per official evaluation rules.
    Returns: (label, badge_color_hex, detail_text)
    
    Allowed distinct labels:
    - Passed official tests
    - Failed functional tests
    - Timeout
    - Generation error
    - Evaluation infrastructure error
    - Pending evaluation
    - Pending generation
    """
    if pred_rec is None:
        return 'Pending generation', '#94A3B8', 'Generation has not been completed for this task.'

    gen_status = pred_rec.get('status', 'unknown')
    if gen_status != 'ok':
        err_msg = pred_rec.get('error', 'Generation error')
        return 'Generation error', '#EF4444', f'Generation failed before evaluation: {err_msg}'

    if eval_rec is None:
        return 'Pending evaluation', '#3B82F6', 'Generation completed successfully; functional evaluation pending.'

    eval_status = eval_rec.get('status')
    if eval_status == 'infrastructure_error':
        return 'Evaluation infrastructure error', '#F59E0B', 'Infrastructure error during test execution environment setup.'

    official_result = eval_rec.get('official_result')
    diagnostic = eval_rec.get('diagnostic', '')

    if official_result == 'Pass':
        return 'Passed official tests', '#10B981', 'All official functional assertion tests passed successfully.'
    elif official_result == 'TimeOut':
        return 'Timeout', '#DC2626', f'Official execution timed out (limit: 60s). Diagnostic: {diagnostic}'
    elif official_result == 'Error':
        if diagnostic == 'generation_error':
            return 'Generation error', '#EF4444', 'Model failed to generate executable code block.'
        return 'Failed functional tests', '#EF4444', f'Functional test failed. Diagnostic: {diagnostic}'
    else:
        return 'Pending evaluation', '#3B82F6', f'Evaluation status: {official_result or "in-progress"}'


def load_ground_truth_body(task_info: Dict[str, Any]) -> str:
    """Extracts the true reference function body from the untouched source code."""
    completion_path = task_info.get('completion_path')
    body_pos = task_info.get('body_position')
    if not completion_path or not body_pos:
        return ""
    
    file_path = SOURCE_CODE_DIR / completion_path
    if not file_path.exists():
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        start_line = max(0, body_pos[0] - 1)
        end_line = min(len(lines), body_pos[1])
        return "".join(lines[start_line:end_line])
    except Exception:
        return ""

