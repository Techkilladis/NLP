"""
Interactive Context Retrieval and Qwen Code Generation Engine.
Implements the dual-method pipeline comparing:
1. Method A: No Retrieval (none) - Zero repository context
2. Method B: DA-RAG-B (darag_b) - Dependency graph retrieval and budget packing
Both evaluated identically using Qwen2.5-Coder-1.5B-Instruct under identical decoding parameters.
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Global model cache for live GPU inference (reused across requests)
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
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=device
        )
        _LIVE_TOKENIZER = tok
        _LIVE_MODEL = model
        return _LIVE_MODEL, _LIVE_TOKENIZER
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        return None, None


def extract_function_signature_and_name(code_prompt: str) -> Tuple[str, str, str]:
    """
    Extracts function name, signature, and docstring from user input code.
    """
    lines = code_prompt.strip().splitlines()
    func_name = "custom_function"
    sig = code_prompt.strip()
    doc = ""

    for line in lines:
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
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
    Builds the official prompt passed to Qwen2.5-Coder-1.5B-Instruct using its chat template.
    """
    try:
        from darag.prompting import build_messages
        messages = build_messages(input_code=input_code, function_name=func_name, context_text=context_code or "")
        _, tok = get_or_load_qwen_model()
        if tok:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass

    # Fallback template matching the official format
    instruction = (
        f"Please complete the {func_name} function in the given Python code.\n"
        f"Return ONLY the body of the function (the indented statements that come after the signature and docstring), "
        f"inside a single ```python code block. Do not repeat the signature or the docstring."
    )

    user_content = ""
    if context_code and context_code.strip():
        user_content += f"The following code from the same repository may be relevant (read-only context):\n```python\n{context_code.strip()}\n```\n\n"

    user_content += f"{instruction}\n\nInput Code:\n```python\n{input_code.strip()}\n```"

    full_prompt = (
        "<|im_start|>system\n"
        "You are an expert Python programmer working inside an existing code repository.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return full_prompt


def extract_code_completion(raw_text: str) -> str:
    """Extracts Python code inside a markdown block or raw body."""
    match = re.search(r"```python\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match_any = re.search(r"```\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match_any:
        return match_any.group(1).strip()
    return raw_text.strip()


def run_single_generation(
    prompt_text: str,
    max_new_tokens: int = 384
) -> Tuple[str, str, int, float, Optional[str]]:
    """
    Runs greedy generation on local Qwen model.
    Returns: (cleaned_completion, raw_output, completion_tokens, latency_s, error)
    """
    model, tokenizer = get_or_load_qwen_model()
    if model is None or tokenizer is None:
        return "", "", 0, 0.0, "Local Qwen model weights could not be loaded into memory."

    import torch
    t0 = time.time()
    try:
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
        completion = extract_code_completion(full_output)
        return completion, full_output, completion_tokens, latency, None
    except Exception as e:
        return "", "", 0, time.time() - t0, f"Generation failed: {str(e)}"


def generate_dual_implementations(
    func_name: str,
    input_code: str,
    darag_context_text: str,
    max_new_tokens: int = 384
) -> Dict[str, Any]:
    """
    Generates implementations under both conditions using the SAME Qwen model and parameters:
    1. Method A: none (0 context)
    2. Method B: darag_b (packed repository context)
    """
    # 1. Prompt for Method A (None / Zero Retrieval)
    prompt_none = build_qwen_chat_prompt(func_name=func_name, input_code=input_code, context_code="")

    # 2. Prompt for Method B (DA-RAG-B)
    prompt_dag = build_qwen_chat_prompt(func_name=func_name, input_code=input_code, context_code=darag_context_text)

    # 3. Generate Method A
    comp_none, raw_none, tok_none, lat_none, err_none = run_single_generation(prompt_none, max_new_tokens)

    # 4. Generate Method B
    comp_dag, raw_dag, tok_dag, lat_dag, err_dag = run_single_generation(prompt_dag, max_new_tokens)

    # Validate syntax of completions
    def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
        if not code or not code.strip():
            return False, "Empty completion"
        try:
            # Wrap in dummy func if it's just indented body
            test_src = code
            if not code.strip().startswith("def ") and not code.strip().startswith("async def "):
                test_src = f"def _dummy():\n{textwrap_indent(code, '    ')}"
            ast.parse(test_src)
            return True, "Valid Python syntax"
        except SyntaxError as se:
            return False, f"SyntaxError at line {se.lineno}: {se.msg}"
        except Exception as e:
            return False, str(e)

    def textwrap_indent(text: str, prefix: str) -> str:
        import textwrap
        return textwrap.indent(textwrap.dedent(text), prefix)

    syn_none, syn_msg_none = check_syntax(comp_none)
    syn_dag, syn_msg_dag = check_syntax(comp_dag)

    return {
        "none": {
            "method": "none",
            "prompt_text": prompt_none,
            "completion": comp_none,
            "raw_output": raw_none,
            "tokens": tok_none,
            "latency": lat_none,
            "error": err_none,
            "syntax_valid": syn_none,
            "syntax_msg": syn_msg_none,
            "context_units": 0,
            "context_tokens": 0
        },
        "darag_b": {
            "method": "darag_b",
            "prompt_text": prompt_dag,
            "completion": comp_dag,
            "raw_output": raw_dag,
            "tokens": tok_dag,
            "latency": lat_dag,
            "error": err_dag,
            "syntax_valid": syn_dag,
            "syntax_msg": syn_msg_dag,
            "context_units": 0,  # Will be populated from context result
            "context_tokens": 0
        }
    }
