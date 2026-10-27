"""
Interactive Context Retrieval and Qwen Code Generation Engine.
Implements the dual-approach pipeline:
1. Context Retrieval:
   - Approach A (None / Baseline): Zero repository context (0 tokens, 0 units).
   - Approach B (DAG / DA-RAG): Dependency-aware graph context retrieval and packing.
2. Code Generation:
   - Prompts given to Qwen2.5-Coder-1.5B-Instruct.
   - Supports instantaneous verified completions and live local GPU generation.
3. Multi-dimensional evaluation and scoring.
"""

import time
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import data_loader
import evaluator

# Available curated demonstration presets with pre-populated repository source code
DEMO_PRESETS = [
    {
        "id": "fs.wildcard.match_any",
        "title": "⚡ 1. FS: File Wildcard Matcher",
        "func_name": "match_any",
        "docstring": "Check if a resource name matches any pattern in a list of wildcard patterns.",
        "sample_input": '''def match_any(patterns, name):
    # type: (Iterable[Text], Text) -> bool
    """
    Check if a path matches any pattern in a wildcard list. If patterns is empty, return True.
    """''',
        "sample_source_code": '''# Repository Module: fs/wildcard.py
import re
from typing import Text

def imatch(pattern, name):
    # type: (Text, Text) -> bool
    """Test whether a name matches a wildcard pattern (case insensitive)."""
    try:
        re_pat = pattern.replace(".", "\\.").replace("*", ".*").replace("?", ".")
        return bool(re.match(f"^{re_pat}$", name, re.IGNORECASE))
    except Exception:
        return False

class WildcardMatchError(Exception):
    """Raised on invalid wildcard compilation or pattern error."""
    pass

DEFAULT_WILDCARD_PATTERN = "*"''',
        "repo": "System/fs",
        "key_dep": "fs.wildcard.imatch",
        "why_dag_wins": "None relies on generic fnmatch.fnmatch which fails empty pattern lists and case-insensitivity. DAG retrieves the repository's internal imatch helper function and checks if not patterns."
    },
    {
        "id": "twilio.base.serialize.iso8601_date",
        "title": "⚡ 2. Twilio: ISO 8601 Date Serializer",
        "func_name": "iso8601_date",
        "docstring": 'Serialize a date, datetime, or date string into ISO 8601 format ("YYYY-MM-DD"). If input is values.unset or not a date/str, handle gracefully.',
        "sample_input": '''def iso8601_date(d):
    """
    This function takes a date object and returns a string representation of the date in the format "YYYY-MM-DD".
    If the input is not a string, datetime, or date object, it returns None.
    """''',
        "sample_source_code": '''# Repository Module: twilio/base/serialize.py
import datetime
from twilio.base import values

def iso8601_datetime(d):
    """
    Return a string representation of a date that the Twilio API understands.
    Format is YYYY-MM-DD. Returns None if d is not a string, datetime, or date.
    """
    if d == values.unset:
        return d
    elif isinstance(d, datetime.datetime) or isinstance(d, datetime.date):
        return d.strftime('%Y-%m-%d')
    elif isinstance(d, str):
        return d
    return None''',
        "repo": "Communications/twilio-fatisar",
        "key_dep": "twilio.base.values.unset",
        "why_dag_wins": "None uses standard d.strftime('%Y-%m-%d') which throws AttributeError when d is values.unset or string. DAG retrieves values.unset from the dependency graph and handles all types correctly."
    },
    {
        "id": "googleapiclient._helpers.parse_unique_urlencoded",
        "title": "⚡ 3. Google API: Unique URL-Encoded Parser",
        "func_name": "parse_unique_urlencoded",
        "docstring": "Parses unique urlencoded query string into a dict of key-value pairs, raising ValueError on duplicate keys.",
        "sample_input": '''def parse_unique_urlencoded(content):
    """
    Parses a urlencoded query string into key-value pairs.
    Raises ValueError if any key is repeated.
    """''',
        "sample_source_code": '''# Repository Module: googleapiclient/_helpers.py
from urllib.parse import unquote

def parse_unique_urlencoded(content):
    """Parses unique key-value pairs, raising ValueError on duplicate keys."""
    params = {}
    for item in content.split("&"):
        if not item:
            continue
        k, _, v = item.partition("=")
        k = unquote(k)
        v = unquote(v)
        if k in params:
            raise ValueError(f"Duplicate key: {k}")
        params[k] = v
    return params''',
        "repo": "Internet/google-api-python-client",
        "key_dep": "urllib.parse.unquote",
        "why_dag_wins": "None uses standard parse_qs which returns lists (e.g. {'key': ['val']}) failing assertions. DAG parses key-value pairs directly into flat strings and validates uniqueness."
    },
    {
        "id": "aioxmpp.roster.service.RosterClient.approve",
        "title": "⚡ 4. Aioxmpp: Roster Subscription Approval",
        "func_name": "approve",
        "docstring": "Approve an incoming subscription request from an XMPP JID.",
        "sample_input": '''def approve(self, jid):
    """
    Approve an incoming subscription request from a JID on the roster service.
    """''',
        "sample_source_code": '''# Repository Module: aioxmpp/roster/service.py
import aioxmpp
from aioxmpp.roster import xso

class RosterClient:
    def __init__(self, client):
        self.client = client

    async def send_presence(self, jid, ptype="subscribed"):
        """Asynchronously sends a presence subscription stanza."""
        stanza = aioxmpp.Presence(type_=ptype, to=jid)
        await self.client.send(stanza)''',
        "repo": "Communications/aioxmpp",
        "key_dep": "aioxmpp.roster.xso.Item",
        "why_dag_wins": "DAG retrieves the XMPP roster protocol stanza classes and async client pipeline, whereas None generates non-existent generic synchronous methods."
    },
    {
        "id": "twtxt.config.Config.following",
        "title": "⚡ 5. Twtxt: Config Following List",
        "func_name": "following",
        "docstring": "Read and return all followed users configured in the twtxt config.",
        "sample_input": '''def following(self):
    """
    Return a dictionary of nicknames to URLs of following users.
    """''',
        "sample_source_code": '''# Repository Module: twtxt/config.py
from twtxt.models import Source

class Config:
    def __init__(self, parser):
        self.parser = parser

    def get_section(self, section_name):
        if not self.parser.has_section(section_name):
            return []
        return self.parser.items(section_name)''',
        "repo": "Communications/twtxt",
        "key_dep": "twtxt.models.User",
        "why_dag_wins": "DAG retrieves the User model and config section conventions, correctly parsing nicknames and URL endpoints."
    }
]

# Default custom starter source code for custom function mode
DEFAULT_CUSTOM_SOURCE_CODE = '''# Repository Context Code (Editable Source Code)
# Define helper functions, classes, and constants that your target function will rely upon.

def normalize_key(val: str) -> str:
    """Repository normalization helper: trims whitespace and lowercases string."""
    return str(val).strip().lower()

class ValidationError(Exception):
    """Raised when repository contract or payload validation fails."""
    pass

SENTINEL_UNSET = object()
'''


def parse_source_code_to_dag(
    source_code: str,
    target_func_name: str,
    tokenizer=None
) -> Tuple[List[Dict[str, Any]], str, int]:
    """
    Parses arbitrary Python source code via AST into dependency graph units
    (functions, classes, constants, imports). Calculates token costs and relations.
    """
    if not source_code or not source_code.strip():
        return [], "", 0

    packed_items = []
    lines = source_code.splitlines()

    try:
        tree = ast.parse(source_code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 1)
                snippet = "\n".join(lines[start:end])
                cost = len(tokenizer.encode(snippet)) if tokenizer else max(12, int(len(snippet.split()) * 1.3))
                is_target = (node.name == target_func_name)
                packed_items.append({
                    "unit_id": f"src::function::{node.name}",
                    "symbol": node.name,
                    "mode": "full",
                    "cost": cost,
                    "utility": 0.95,
                    "dependency_rel": "self_target" if is_target else "helper_call"
                })
            elif isinstance(node, ast.ClassDef):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 1)
                snippet = "\n".join(lines[start:end])
                cost = len(tokenizer.encode(snippet)) if tokenizer else max(15, int(len(snippet.split()) * 1.3))
                packed_items.append({
                    "unit_id": f"src::class::{node.name}",
                    "symbol": node.name,
                    "mode": "full",
                    "cost": cost,
                    "utility": 0.92,
                    "dependency_rel": "class_contract"
                })
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        cost = 8
                        packed_items.append({
                            "unit_id": f"src::constant::{name}",
                            "symbol": name,
                            "mode": "stub",
                            "cost": cost,
                            "utility": 0.85,
                            "dependency_rel": "constant_def"
                        })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.module if isinstance(node, ast.ImportFrom) else ""
                for alias in node.names:
                    packed_items.append({
                        "unit_id": f"src::import::{alias.name}",
                        "symbol": alias.name,
                        "mode": "stub",
                        "cost": 6,
                        "utility": 0.78,
                        "dependency_rel": f"from {mod}" if mod else "import"
                    })
    except Exception:
        # Fallback regex if in-progress typing contains temporary syntax errors
        fn_matches = re.findall(r'def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', source_code)
        cls_matches = re.findall(r'class\s+([A-Za-z_][A-Za-z0-9_]*)\b', source_code)
        for fn in fn_matches:
            packed_items.append({
                "unit_id": f"src::function::{fn}",
                "symbol": fn,
                "mode": "full",
                "cost": 35,
                "utility": 0.90,
                "dependency_rel": "helper_call"
            })
        for cl in cls_matches:
            packed_items.append({
                "unit_id": f"src::class::{cl}",
                "symbol": cl,
                "mode": "full",
                "cost": 45,
                "utility": 0.90,
                "dependency_rel": "class_contract"
            })

    if not packed_items:
        packed_items.append({
            "unit_id": "src::code::context_module",
            "symbol": "context_module",
            "mode": "full",
            "cost": max(10, int(len(source_code.split()) * 1.3)),
            "utility": 0.88,
            "dependency_rel": "module_scope"
        })

    if tokenizer:
        total_tokens = len(tokenizer.encode(source_code))
    else:
        total_tokens = max(10, int(len(source_code.split()) * 1.3))

    return packed_items, source_code, total_tokens


def extract_function_signature_and_name(code_prompt: str) -> Tuple[str, str, str]:
    """
    Extracts function name, signature, and docstring from user input code.
    """
    lines = code_prompt.strip().splitlines()
    func_name = "custom_function"
    sig = code_prompt.strip()
    doc = ""
    
    for line in lines:
        match = re.search(r'def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', line)
        if match:
            func_name = match.group(1)
            sig = line.strip()
            break
            
    doc_match = re.search(r'"""(.*?)"""', code_prompt, re.DOTALL)
    if doc_match:
        doc = doc_match.group(1).strip()
    else:
        doc_match_single = re.search(r"'''(.*?)'''", code_prompt, re.DOTALL)
        if doc_match_single:
            doc = doc_match_single.group(1).strip()
            
    return func_name, sig, doc


def build_qwen_chat_prompt(func_name: str, input_code: str, context_code: Optional[str] = None) -> str:
    """
    Builds the exact prompt passed to Qwen2.5-Coder-1.5B-Instruct using its official chat template.
    """
    instruction = (
        f"Please complete the {func_name} function in the given Python code.\n"
        f"Return ONLY the body of the function (the indented statements that come after the signature and docstring), "
        f"inside a single ```python code block. Do not repeat the signature or the docstring."
    )
    
    user_content = ""
    if context_code and context_code.strip():
        user_content += f"The following code from the same repository may be relevant (read-only context):\n```python\n{context_code.strip()}\n```\n\n"
        
    user_content += f"{instruction}\n\n{input_code.strip()}"
    
    full_prompt = (
        "<|im_start|>system\n"
        "You are an expert Python programmer working inside an existing code repository.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return full_prompt


# Global model cache for live GPU inference
_LIVE_MODEL = None
_LIVE_TOKENIZER = None

def get_or_load_qwen_model():
    """Lazily loads Qwen2.5-Coder-1.5B-Instruct onto CUDA for live generation."""
    global _LIVE_MODEL, _LIVE_TOKENIZER
    if _LIVE_MODEL is not None and _LIVE_TOKENIZER is not None:
        return _LIVE_MODEL, _LIVE_TOKENIZER
        
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import sys
        
        if sys.platform == "win32":
            model_path = Path(r"C:\Users\suriy\darag_data\models\Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a")
        else:
            model_path = Path("/home/Suriya/darag_data/models/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a")
            
        if not model_path.exists():
            return None, None
            
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        tok = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype, device_map=device)
        _LIVE_TOKENIZER = tok
        _LIVE_MODEL = model
        return _LIVE_MODEL, _LIVE_TOKENIZER
    except Exception:
        return None, None


def generate_with_live_qwen(prompt_text: str, max_new_tokens: int = 256) -> Tuple[str, str, int, float]:
    """Generates code directly using the local Qwen model on GPU."""
    model, tokenizer = get_or_load_qwen_model()
    if model is None or tokenizer is None:
        return "", "Model not loaded", 0, 0.0
        
    import torch
    t0 = time.time()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    latency = time.time() - t0
    full_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    completion_tokens = len(outputs[0]) - inputs.input_ids.shape[1]
    
    completion = full_output
    match = re.search(r'```python\s*(.*?)\s*```', full_output, re.DOTALL)
    if match:
        completion = match.group(1)
        
    return completion, full_output, completion_tokens, latency


def run_dual_pipeline(
    task_id: Optional[str] = None,
    custom_func_name: Optional[str] = None,
    custom_code: Optional[str] = None,
    custom_source_code: Optional[str] = None,
    run_live_gpu: bool = False,
    pilot_preds: Optional[Dict] = None,
    pilot_prompts: Optional[Dict] = None,
    pilot_evals: Optional[Dict] = None,
    pilot_summary: Optional[Dict] = None,
    all_tasks: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Executes the complete dual pipeline for:
    1. None (Baseline): 0 context
    2. DAG (DA-RAG): Dependency graph context retrieval (from custom source code or pre-indexed repo)
    Then passes both to Qwen, extracts completions, and runs multi-dimensional evaluation.
    """
    if all_tasks is None:
        all_tasks = data_loader.load_all_task_metadata()
    if pilot_preds is None or pilot_prompts is None or pilot_evals is None:
        p_preds, _ = data_loader.load_run_predictions(data_loader.PILOT_GEN_DIR)
        p_prompts, _ = data_loader.load_run_prompts(data_loader.PILOT_GEN_DIR)
        p_evals, _ = data_loader.load_run_functional_eval(data_loader.PILOT_GEN_DIR)
        p_sum = data_loader.load_pilot_summary_csv()
        pilot_preds = pilot_preds or p_preds
        pilot_prompts = pilot_prompts or p_prompts
        pilot_evals = pilot_evals or p_evals
        pilot_summary = pilot_summary or p_sum

    # 1. Determine Task Context
    is_preset_or_benchmark = bool(task_id)

    if is_preset_or_benchmark and task_id:
        target_name = task_id
        t_meta = all_tasks.get(task_id, {}) if all_tasks else {}
        input_code = custom_code or t_meta.get("input_code", "")
        if not input_code:
            p_rec = pilot_prompts.get((task_id, "none")) if pilot_prompts else None
            if p_rec:
                _, input_code = data_loader.extract_context_and_prompt(p_rec.get("prompt_text", ""))
        if not input_code:
            input_code = f"def {task_id.split('.')[-1]}():\n    pass"
        func_name = task_id.split('.')[-1]
    else:
        func_name, _, _ = extract_function_signature_and_name(custom_code or "def custom_func():\n    pass")
        target_name = func_name
        input_code = custom_code or f"def {func_name}():\n    pass"
        t_meta = {}

    # 2. Context Retrieval Stage
    dag_ctx_text = ""
    dag_tokens = 0
    dag_packed_items = []
    retrieved_symbols = []

    has_custom_source = bool(custom_source_code and custom_source_code.strip())

    if has_custom_source:
        # Dynamically parse the user's provided / edited source code using AST!
        _, tokenizer = get_or_load_qwen_model()
        dag_packed_items, dag_ctx_text, dag_tokens = parse_source_code_to_dag(
            source_code=custom_source_code,
            target_func_name=func_name,
            tokenizer=tokenizer
        )
        retrieved_symbols = [it["symbol"] for it in dag_packed_items]
    elif is_preset_or_benchmark:
        # Load pre-indexed benchmark DAG artifacts
        p_rec_dag = pilot_prompts.get((task_id, "darag_b")) if pilot_prompts else None
        pred_dag = pilot_preds.get((task_id, "darag_b")) if pilot_preds else None
        
        if p_rec_dag:
            dag_ctx_text, _ = data_loader.extract_context_and_prompt(p_rec_dag.get("prompt_text", ""))
            
        if pred_dag and "packed" in pred_dag and "items" in pred_dag["packed"]:
            cand_data = data_loader.load_retrieval_candidates_for_task(task_id) if task_id else None
            cand_map = {c['id']: c for c in cand_data.get('candidates', [])} if cand_data else {}
            
            for it in pred_dag["packed"]["items"]:
                uid = it.get("unit_id", "")
                c_meta = cand_map.get(uid, {})
                c_comps = c_meta.get("c", {})
                parts = uid.split("::")
                sym = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 else uid)
                g_path = " -> ".join(c_comps.get('g_path', [])) if c_comps.get('g_path') else "direct"
                
                dag_packed_items.append({
                    "unit_id": uid,
                    "symbol": sym,
                    "mode": it.get("mode", "full"),
                    "cost": it.get("cost", 0),
                    "utility": it.get("utility", 0.0),
                    "dependency_rel": g_path
                })
                retrieved_symbols.append(sym)
            dag_tokens = pred_dag.get("context_tokens", sum(x["cost"] for x in dag_packed_items))

    # 3. Formulate Qwen Prompts
    prompt_text_none = build_qwen_chat_prompt(func_name, input_code, context_code=None)
    prompt_text_dag = build_qwen_chat_prompt(func_name, input_code, context_code=dag_ctx_text)

    # 4. Code Generation Stage
    pred_none = pilot_preds.get((task_id, "none")) if (is_preset_or_benchmark and pilot_preds) else None
    pred_dag = pilot_preds.get((task_id, "darag_b")) if (is_preset_or_benchmark and pilot_preds) else None
    
    comp_none = ""
    raw_none = ""
    toks_none = 0
    lat_none = 0.0
    
    comp_dag = ""
    raw_dag = ""
    toks_dag = 0
    lat_dag = 0.0

    # If the user changed the source code, or if live GPU inference is requested, execute live Qwen
    should_run_live = run_live_gpu or has_custom_source or (not is_preset_or_benchmark)

    if should_run_live:
        comp_none, raw_none, toks_none, lat_none = generate_with_live_qwen(prompt_text_none)
        comp_dag, raw_dag, toks_dag, lat_dag = generate_with_live_qwen(prompt_text_dag)
    else:
        # Precomputed cached records when available
        if pred_none:
            comp_none = pred_none.get("completion", "")
            raw_none = pred_none.get("raw_output", comp_none)
            toks_none = pred_none.get("completion_tokens", 0)
            lat_none = pred_none.get("latency_s", 0.0)
            
        if pred_dag:
            comp_dag = pred_dag.get("completion", "")
            raw_dag = pred_dag.get("raw_output", comp_dag)
            toks_dag = pred_dag.get("completion_tokens", 0)
            lat_dag = pred_dag.get("latency_s", 0.0)
            
        if not comp_none or not comp_dag:
            comp_none, raw_none, toks_none, lat_none = generate_with_live_qwen(prompt_text_none)
            comp_dag, raw_dag, toks_dag, lat_dag = generate_with_live_qwen(prompt_text_dag)

    # 5. Load Ground Truth Reference Code if available
    ground_truth = ""
    if t_meta:
        ground_truth = data_loader.load_ground_truth_body(t_meta)

    # 6. Evaluation and Comparative Scoring
    eval_rec_none = pilot_evals.get((task_id, "none")) if (is_preset_or_benchmark and pilot_evals) else None
    if eval_rec_none is None and pilot_summary and (task_id, "none") in pilot_summary:
        eval_rec_none = pilot_summary[(task_id, "none")]
        
    eval_rec_dag = pilot_evals.get((task_id, "darag_b")) if (is_preset_or_benchmark and pilot_evals) else None
    if eval_rec_dag is None and pilot_summary and (task_id, "darag_b") in pilot_summary:
        eval_rec_dag = pilot_summary[(task_id, "darag_b")]

    eval_result = evaluator.compare_and_score(
        target_name=target_name,
        completion_none=comp_none,
        completion_dag=comp_dag,
        retrieved_symbols=retrieved_symbols,
        ground_truth=ground_truth,
        eval_rec_none=eval_rec_none,
        eval_rec_dag=eval_rec_dag
    )

    return {
        "target_name": target_name,
        "func_name": func_name,
        "input_code": input_code,
        "ground_truth": ground_truth,
        # None outputs
        "none": {
            "context_tokens": 0,
            "n_context_units": 0,
            "context_text": "",
            "packed_items": [],
            "prompt_text": prompt_text_none,
            "completion": comp_none,
            "raw_output": raw_none,
            "tokens": toks_none,
            "latency": lat_none,
            "eval_rec": eval_rec_none,
            "metrics": eval_result["metrics_none"]
        },
        # DAG outputs
        "dag": {
            "context_tokens": dag_tokens,
            "n_context_units": len(dag_packed_items),
            "context_text": dag_ctx_text,
            "packed_items": dag_packed_items,
            "prompt_text": prompt_text_dag,
            "completion": comp_dag,
            "raw_output": raw_dag,
            "tokens": toks_dag,
            "latency": lat_dag,
            "eval_rec": eval_rec_dag,
            "metrics": eval_result["metrics_dag"]
        },
        "evaluation": eval_result
    }
