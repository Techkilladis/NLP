"""
Evaluation and scoring engine for DA-RAG Code Generation Test Bench.
Evaluates both None and DAG completions across:
1. Syntax & AST Validity
2. Context & Dependency Utilization (detection of repo-specific symbols/helpers)
3. Interface & Contract Conformance
4. Ground Truth / Reference Similarity (when available)
5. Composite Score (0-100%) and Winner Determination with narrative explanation.
"""

import ast
import re
from typing import Dict, List, Optional, Any, Tuple


def check_ast_validity(code_str: str) -> Tuple[bool, str]:
    """Checks if the code snippet is syntactically valid Python."""
    if not code_str or not code_str.strip():
        return False, "Empty code snippet"
    
    clean_code = code_str.strip()
    
    # Try parsing directly
    try:
        ast.parse(clean_code)
        return True, "Valid Python syntax"
    except SyntaxError:
        pass

    # Try indenting inside a dummy function (in case it is an unindented function body)
    indented = "\n".join("    " + line for line in clean_code.splitlines())
    wrapped = f"def _dummy_eval_func():\n{indented}\n"
    try:
        ast.parse(wrapped)
        return True, "Valid function body syntax"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def extract_ast_identifiers(code_str: str) -> set:
    """Extracts all variable, function call, and attribute names from Python code."""
    names = set()
    clean_code = code_str.strip()
    
    # Try parsing
    tree = None
    try:
        tree = ast.parse(clean_code)
    except SyntaxError:
        indented = "\n".join("    " + line for line in clean_code.splitlines())
        wrapped = f"def _dummy_eval_func():\n{indented}\n"
        try:
            tree = ast.parse(wrapped)
        except SyntaxError:
            # Fallback to regex word extraction
            return set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', code_str))
            
    if tree:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.FunctionDef):
                names.add(node.name)
            elif isinstance(node, ast.alias):
                names.add(node.name)
    return names


def compute_token_similarity(code_a: str, code_b: str) -> float:
    """Computes token-level Jaccard similarity and character sequence similarity."""
    if not code_a or not code_b:
        return 0.0
    
    tokens_a = re.findall(r'\b\w+\b|[^\w\s]', code_a)
    tokens_b = re.findall(r'\b\w+\b|[^\w\s]', code_b)
    
    if not tokens_a or not tokens_b:
        return 0.0
        
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = (intersection / union) if union > 0 else 0.0
    
    # Also check exact match
    if code_a.strip() == code_b.strip():
        return 1.0
        
    return round(jaccard, 4)


def evaluate_completion(
    completion: str,
    target_name: str,
    retrieved_symbols: List[str],
    ground_truth: Optional[str] = None,
    official_result: Optional[str] = None,
    is_dag: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a single model completion.
    Returns metrics dictionary with component scores and diagnostics.
    """
    if not completion or not completion.strip():
        return {
            "valid": False,
            "syntax_score": 0.0,
            "dependency_score": 0.0,
            "interface_score": 0.0,
            "similarity_score": 0.0,
            "composite_score": 0.0,
            "symbols_used": [],
            "symbols_missed": retrieved_symbols,
            "diagnostic": "Empty completion"
        }
    
    # 1. Syntax Score (0 - 100)
    is_valid_syntax, syntax_msg = check_ast_validity(completion)
    syntax_score = 100.0 if is_valid_syntax else 20.0
    
    # 2. Extract identifiers
    identifiers = extract_ast_identifiers(completion)
    
    # 3. Dependency Utilization Score (0 - 100)
    symbols_used = []
    symbols_missed = []
    
    clean_target = target_name.split('.')[-1]
    
    for sym in retrieved_symbols:
        short_sym = sym.split('::')[-1].split('.')[-1]
        if short_sym and short_sym != clean_target:
            if short_sym in identifiers or short_sym in completion:
                symbols_used.append(short_sym)
            else:
                symbols_missed.append(short_sym)
    
    if retrieved_symbols:
        dep_ratio = len(symbols_used) / max(len(symbols_used) + len(symbols_missed), 1)
        dep_score = round(dep_ratio * 100.0, 1)
        # Give partial credit if valid syntax
        if dep_score == 0 and is_valid_syntax:
            dep_score = 25.0
    else:
        # For none or tasks with no external dependencies
        dep_score = 50.0 if is_valid_syntax else 10.0
        
    # 4. Interface Conformance Score (0 - 100)
    # Checks for return statements, no syntax fence repeats, reasonable length
    interface_score = 70.0
    if "return " in completion or "yield " in completion:
        interface_score += 20.0
    if completion.strip().startswith("def "):
        interface_score -= 20.0  # Should be body only
    if "```" in completion:
        interface_score -= 15.0  # Stray markdown fences
    interface_score = max(0.0, min(100.0, interface_score))
    
    # 5. Ground Truth / Reference Similarity (0 - 100)
    sim_score = 0.0
    if ground_truth:
        sim_val = compute_token_similarity(completion, ground_truth)
        sim_score = round(sim_val * 100.0, 1)
    elif official_result == "Pass":
        sim_score = 95.0
    elif official_result == "Error":
        sim_score = 30.0
    else:
        sim_score = 50.0
        
    # 6. Functional Test Boost / Penalty if official test result is known
    test_factor = 1.0
    if official_result == "Pass":
        test_factor = 1.2
    elif official_result in ("Error", "TimeOut"):
        test_factor = 0.85

    # 7. Weighted Composite Score
    # 30% Functional / Ground Truth, 30% Dependency Utilization, 25% Syntax, 15% Interface
    composite = (
        0.30 * sim_score +
        0.30 * dep_score +
        0.25 * syntax_score +
        0.15 * interface_score
    ) * test_factor
    
    composite = round(max(0.0, min(100.0, composite)), 1)
    
    return {
        "valid": is_valid_syntax,
        "syntax_msg": syntax_msg,
        "syntax_score": round(syntax_score, 1),
        "dependency_score": round(dep_score, 1),
        "interface_score": round(interface_score, 1),
        "similarity_score": round(sim_score, 1),
        "composite_score": composite,
        "symbols_used": symbols_used,
        "symbols_missed": symbols_missed,
        "diagnostic": syntax_msg
    }


def compare_and_score(
    target_name: str,
    completion_none: str,
    completion_dag: str,
    retrieved_symbols: List[str],
    ground_truth: Optional[str] = None,
    eval_rec_none: Optional[Dict[str, Any]] = None,
    eval_rec_dag: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compares None vs DAG completions, scores both, and determines the winner with narrative explanation.
    """
    res_none = eval_rec_none.get("official_result") if eval_rec_none else None
    res_dag = eval_rec_dag.get("official_result") if eval_rec_dag else None
    
    eval_none = evaluate_completion(
        completion=completion_none,
        target_name=target_name,
        retrieved_symbols=retrieved_symbols,
        ground_truth=ground_truth,
        official_result=res_none,
        is_dag=False
    )
    
    eval_dag = evaluate_completion(
        completion=completion_dag,
        target_name=target_name,
        retrieved_symbols=retrieved_symbols,
        ground_truth=ground_truth,
        official_result=res_dag,
        is_dag=True
    )
    
    score_none = eval_none["composite_score"]
    score_dag = eval_dag["composite_score"]
    delta = round(score_dag - score_none, 1)
    
    # Determine winner
    if score_dag > score_none:
        winner = "dag"
        winner_label = "🏆 DAG (DA-RAG) Model"
        delta_str = f"+{delta}% Advantage"
    elif score_none > score_dag:
        winner = "none"
        winner_label = "🏆 Baseline (None) Model"
        delta_str = f"+{-delta}% Advantage"
    else:
        winner = "tie"
        winner_label = "🤝 Tied Performance"
        delta_str = "Equal Score"
        
    # Generate Narrative Explanation
    reasons = []
    if eval_dag["symbols_used"]:
        used_list = ", ".join(f"`{s}`" for s in eval_dag["symbols_used"][:4])
        reasons.append(f"DAG correctly integrated repository-specific dependencies ({used_list}) retrieved from the code dependency graph.")
    
    if not eval_none["symbols_used"] and retrieved_symbols:
        reasons.append("Baseline (None) lacked repository context and attempted generic standard library calls, missing project conventions.")
        
    if eval_dag["similarity_score"] > eval_none["similarity_score"]:
        reasons.append(f"DAG achieved higher semantic alignment with the reference implementation ({eval_dag['similarity_score']}% vs {eval_none['similarity_score']}%).")
        
    if res_dag == "Pass" and res_none != "Pass":
        reasons.append("Official test suite confirmation: DAG passed all functional unit tests whereas Baseline None failed.")
    elif res_dag == "Pass":
        reasons.append("DAG successfully passed all functional assertions in the project test suite.")
        
    if not reasons:
        reasons.append("Both models generated functional Python syntax; score differences reflect interface cleanliness and context relevance.")
        
    narrative = " ".join(reasons)
    
    return {
        "winner": winner,
        "winner_label": winner_label,
        "delta": delta,
        "delta_str": delta_str,
        "score_none": score_none,
        "score_dag": score_dag,
        "eval_none": eval_none,
        "eval_dag": eval_dag,
        "narrative": narrative,
        "key_reasons": reasons
    }
