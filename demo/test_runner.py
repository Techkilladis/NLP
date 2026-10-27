"""
Isolated Test Execution Engine for Repository-Level Code Generation Test Bench.
Runs tests strictly in disposable temporary copies of the selected repository,
leaving the user's original workspace completely untouched.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from repo_manager import normalize_repo_path, find_matching_test_file


class TestExecutionResult:
    def __init__(
        self,
        method: str,                 # 'none' or 'darag_b'
        status: str,                 # 'PASS', 'FAIL', 'TIMEOUT', 'INFRASTRUCTURE ERROR', 'GENERATION ERROR', 'NOT RUN'
        exit_code: Optional[int],
        duration_s: float,
        tests_passed: int,
        tests_failed: int,
        failed_test_names: List[str],
        stdout: str,
        stderr: str,
        syntax_valid: bool,
        syntax_error: Optional[str] = None,
        command_executed: str = "",
        error_details: str = "",
        inserted_code_diff: str = ""
    ):
        self.method = method
        self.status = status
        self.exit_code = exit_code
        self.duration_s = duration_s
        self.tests_passed = tests_passed
        self.tests_failed = tests_failed
        self.failed_test_names = failed_test_names
        self.stdout = stdout
        self.stderr = stderr
        self.syntax_valid = syntax_valid
        self.syntax_error = syntax_error
        self.command_executed = command_executed
        self.error_details = error_details
        self.inserted_code_diff = inserted_code_diff

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 2),
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "failed_test_names": self.failed_test_names,
            "syntax_valid": self.syntax_valid,
            "syntax_error": self.syntax_error,
            "command_executed": self.command_executed,
            "error_details": self.error_details
        }


def detect_best_python_interpreter() -> str:
    """Finds the best available Python interpreter in the WSL environment."""
    candidates = [
        "/home/Suriya/darag_data/eval_env/bin/python",
        "/home/Suriya/.venvs/darag/bin/python",
        sys.executable
    ]
    for c in candidates:
        if Path(c).exists() and os.access(c, os.X_OK):
            return c
    return sys.executable


def adjust_code_indent(code: str, target_indent: int) -> str:
    """
    Safely dedents any code block preserving relative line indentation,
    then indents the entire block to target_indent spaces.
    """
    # Prepend newline so textwrap.dedent evaluates line 1's indentation as part of common indent
    raw = code.rstrip()
    if not raw.startswith("\n"):
        raw = "\n" + raw
    dedented = textwrap.dedent(raw).strip()
    return textwrap.indent(dedented, " " * target_indent)


def insert_function_implementation(
    original_file_text: str,
    func_name: str,
    generated_completion: str,
    target_signature: Optional[str] = None
) -> Tuple[str, bool, Optional[str], str]:
    """
    Safely inserts the generated function body or definition into the target file text.
    Returns: (modified_text, syntax_valid, syntax_error_msg, diff_summary)
    """
    completion_clean = generated_completion.strip()
    if not completion_clean or completion_clean == "pass":
        return original_file_text, False, "Generated code is empty or single 'pass'", ""

    lines = original_file_text.splitlines(keepends=True)
    clean_lines = original_file_text.splitlines()

    # Try AST parse to find existing function location
    try:
        tree = ast.parse(original_file_text)
    except Exception:
        tree = None

    target_node = None
    if tree:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                target_node = node
                break
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == func_name:
                        target_node = sub
                        break
                if target_node:
                    break

    # Determine whether the generated completion is a full `def` or just the indented body
    is_full_def = bool(re.match(r"^(async\s+)?def\s+" + re.escape(func_name) + r"\b", completion_clean.lstrip()))

    modified_text = ""
    diff_summary = ""

    if target_node:
        start_line = target_node.lineno  # 1-indexed
        end_line = getattr(target_node, "end_lineno", start_line)

        # Indentation of the function
        fn_line = clean_lines[start_line - 1]
        base_indent = len(fn_line) - len(fn_line.lstrip())

        if is_full_def:
            # Replace the entire function definition
            indented_code = adjust_code_indent(generated_completion, base_indent)
            pre = "".join(lines[:start_line - 1])
            post = "".join(lines[end_line:])
            modified_text = pre + indented_code + "\n" + post
            diff_summary = f"Replaced definition of {func_name} (lines {start_line}-{end_line})"
        else:
            # Replace only the function body (after docstring/signature)
            body_stmts = target_node.body
            # Skip docstring if present
            body_start_line = body_stmts[0].lineno
            if isinstance(body_stmts[0], ast.Expr) and isinstance(getattr(body_stmts[0], "value", None), (ast.Str, ast.Constant)):
                if len(body_stmts) > 1:
                    body_start_line = body_stmts[1].lineno
                else:
                    body_start_line = getattr(body_stmts[0], "end_lineno", body_stmts[0].lineno) + 1

            body_indent = base_indent + 4
            indented_body = adjust_code_indent(generated_completion, body_indent)
            pre = "".join(lines[:body_start_line - 1])
            post = "".join(lines[end_line:])
            modified_text = pre + indented_body + "\n" + post
            diff_summary = f"Inserted generated body into {func_name} (lines {body_start_line}-{end_line})"
    else:
        # Function does not exist in target file: append at the end
        if is_full_def:
            full_fn = adjust_code_indent(generated_completion, 0)
        else:
            sig = target_signature.strip() if target_signature else f"def {func_name}():"
            indented_body = adjust_code_indent(generated_completion, 4)
            full_fn = f"{sig}\n{indented_body}\n"

        modified_text = original_file_text.rstrip() + "\n\n\n" + full_fn + "\n"
        diff_summary = f"Appended new function {func_name} to end of file"

    # Validate syntax of modified file
    try:
        ast.parse(modified_text)
        return modified_text, True, None, diff_summary
    except SyntaxError as syn_err:
        err_msg = f"SyntaxError at line {syn_err.lineno}: {syn_err.msg}"
        return modified_text, False, err_msg, diff_summary
    except Exception as ex:
        return modified_text, False, str(ex), diff_summary


def parse_pytest_output(stdout: str, stderr: str, exit_code: int) -> Tuple[str, int, int, List[str], str]:
    """
    Parses pytest output to extract test status, passed/failed counts, and failed test names.
    """
    combined = stdout + "\n" + stderr

    # Infrastructure error patterns
    infra_patterns = [
        r"ModuleNotFoundError:\s*No module named\s*['\"]([^'\"]+)['\"]",
        r"ImportError:\s*(.*)",
        r"invalid command\s*['\"]pytest['\"]",
        r"pytest:\s*command not found",
        r"pkg_resources\.DistributionNotFound:\s*(.*)",
        r"ERROR\s+collecting\s+(.*)",
        r"UsageError:\s*(.*)"
    ]
    for pat in infra_patterns:
        m = re.search(pat, combined)
        if m:
            return "INFRASTRUCTURE ERROR", 0, 0, [], f"Environment/Dependency Error: {m.group(0)}"

    # Check for passed / failed counts
    # e.g. "=== 3 passed, 1 failed in 0.12s ===" or "=== 2 passed in 0.05s ===" or "=== 1 failed in 0.02s ==="
    passed_cnt = 0
    failed_cnt = 0

    m_pass = re.search(r"(\d+)\s+passed", combined)
    if m_pass:
        passed_cnt = int(m_pass.group(1))

    m_fail = re.search(r"(\d+)\s+failed", combined)
    if m_fail:
        failed_cnt = int(m_fail.group(1))

    m_err = re.search(r"(\d+)\s+error", combined)
    if m_err:
        failed_cnt += int(m_err.group(1))

    # Failed test names
    failed_names = []
    # Lines starting with FAILED tests/test_foo.py::test_bar
    for line in combined.splitlines():
        line_s = line.strip()
        if line_s.startswith("FAILED "):
            fn = line_s.replace("FAILED ", "").split(" - ")[0].strip()
            failed_names.append(fn)

    if exit_code == 0 and (passed_cnt > 0 or "passed" in combined):
        return "PASS", passed_cnt, failed_cnt, failed_names, ""

    if failed_cnt > 0 or failed_names:
        return "FAIL", passed_cnt, failed_cnt, failed_names, f"{failed_cnt} test(s) failed."

    if exit_code != 0:
        # Check if pytest found no tests or failed to run
        if "collected 0 items" in combined:
            return "INFRASTRUCTURE ERROR", 0, 0, [], "Pytest collected 0 items (check test file/selector)."
        return "FAIL", passed_cnt, max(1, failed_cnt), failed_names, f"Test command exited with code {exit_code}."

    return "PASS", passed_cnt, 0, [], ""


def run_test_in_isolated_copy(
    repo_root: str,
    target_file_rel: str,
    func_name: str,
    generated_code: str,
    test_command: str,
    method: str = "method",
    timeout_s: float = 30.0,
    python_interpreter: Optional[str] = None,
    target_signature: Optional[str] = None
) -> TestExecutionResult:
    """
    Creates an isolated temporary copy of repo_root, patches target_file_rel with generated_code,
    runs test_command, and cleans up. The original repository is strictly untouched.
    """
    t0 = time.time()
    # 1. Check generated code
    if not generated_code or not generated_code.strip():
        return TestExecutionResult(
            method=method,
            status="GENERATION ERROR",
            exit_code=None,
            duration_s=0.0,
            tests_passed=0,
            tests_failed=0,
            failed_test_names=[],
            stdout="",
            stderr="",
            syntax_valid=False,
            error_details="Model produced an empty completion."
        )

    # 2. Check repository directory (normalizing Windows drive letters to POSIX if in WSL)
    norm_repo_root = normalize_repo_path(repo_root) if repo_root else ""
    repo_p = Path(norm_repo_root) if norm_repo_root else Path(repo_root)
    if not repo_p.exists() or not repo_p.is_dir():
        # Fallback check if raw path exists directly on current filesystem
        raw_p = Path(repo_root)
        if raw_p.exists() and raw_p.is_dir():
            repo_p = raw_p
        else:
            return TestExecutionResult(
                method=method,
                status="INFRASTRUCTURE ERROR",
                exit_code=None,
                duration_s=0.0,
                tests_passed=0,
                tests_failed=0,
                failed_test_names=[],
                stdout="",
                stderr="",
                syntax_valid=False,
                error_details=f"Repository directory does not exist: {repo_root} (resolved: {norm_repo_root})"
            )

    # 2. Create isolated temp directory
    temp_dir = tempfile.mkdtemp(prefix=f"darag_eval_{method}_")

    try:
        # Copy repo into temp_dir (excluding heavy/unneeded files, virtualenvs, datasets, spreadsheets)
        ignore_patterns = shutil.ignore_patterns(
            ".git", ".venv", "venv", "env", "__pycache__",
            ".pytest_cache", ".tox", "build", "dist", "*.egg-info",
            "node_modules", ".claude", ".codex", ".playwright-mcp",
            "*.xlsx", "*.xls", "*.csv", "*.tar*", "*.zip", "*.gz",
            "xls files", "archive", "*.db", "*.sqlite3"
        )
        dest_p = Path(temp_dir) / repo_p.name
        shutil.copytree(repo_p, dest_p, ignore=ignore_patterns, symlinks=True)

        target_in_copy = dest_p / target_file_rel
        if not target_in_copy.exists():
            return TestExecutionResult(
                method=method,
                status="INFRASTRUCTURE ERROR",
                exit_code=None,
                duration_s=time.time() - t0,
                tests_passed=0,
                tests_failed=0,
                failed_test_names=[],
                stdout="",
                stderr="",
                syntax_valid=False,
                error_details=f"Target file not found in repository: {target_file_rel}"
            )

        # 3. Read original target file and insert generated code
        orig_content = target_in_copy.read_text(encoding="utf-8", errors="replace")
        modified_content, syntax_ok, syn_err, diff_info = insert_function_implementation(
            orig_content,
            func_name,
            generated_code,
            target_signature
        )

        target_in_copy.write_text(modified_content, encoding="utf-8")

        if not syntax_ok:
            return TestExecutionResult(
                method=method,
                status="GENERATION ERROR",
                exit_code=None,
                duration_s=time.time() - t0,
                tests_passed=0,
                tests_failed=0,
                failed_test_names=[],
                stdout="",
                stderr="",
                syntax_valid=False,
                syntax_error=syn_err,
                error_details=f"Generated code produced Python syntax error: {syn_err}",
                inserted_code_diff=diff_info
            )

        # 4. Prepare test execution
        py_exe = python_interpreter or detect_best_python_interpreter()
        
        # If test_command contains placeholders like <matching_test_file> or <test_file>
        cmd_str = test_command.strip()
        if "<" in cmd_str and ">" in cmd_str:
            matched = find_matching_test_file(str(repo_p), target_file_rel)
            if not matched:
                if (dest_p / "tests").is_dir():
                    matched = "tests/"
                elif (dest_p / "test").is_dir():
                    matched = "test/"
                else:
                    matched = ""
            cmd_str = re.sub(r"<[^>]+>", matched, cmd_str).strip()
            cmd_str = re.sub(r"\s+", " ", cmd_str)

        # If test_command starts with 'pytest', replace with '<py_exe> -m pytest'
        if cmd_str.startswith("pytest"):
            full_cmd = f"{py_exe} -m {cmd_str}"
        elif cmd_str.startswith("python "):
            full_cmd = f"{py_exe} {cmd_str[7:]}"
        else:
            full_cmd = cmd_str

        # Environment variables
        env = dict(os.environ)
        env["PYTHONPATH"] = f".:{str(dest_p)}:" + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("NLTK_DATA", "/home/Suriya/darag_data/nltk_data")

        # 5. Execute command with bounded timeout
        try:
            proc = subprocess.Popen(
                full_cmd,
                shell=True,
                cwd=str(dest_p),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            stdout, stderr = proc.communicate(timeout=timeout_s)
            exit_code = proc.returncode
            duration = time.time() - t0
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return TestExecutionResult(
                method=method,
                status="TIMEOUT",
                exit_code=None,
                duration_s=timeout_s,
                tests_passed=0,
                tests_failed=0,
                failed_test_names=[],
                stdout=stdout or "",
                stderr=stderr or "",
                syntax_valid=True,
                command_executed=full_cmd,
                error_details=f"Execution timed out after {timeout_s}s",
                inserted_code_diff=diff_info
            )
        except Exception as ex:
            return TestExecutionResult(
                method=method,
                status="INFRASTRUCTURE ERROR",
                exit_code=None,
                duration_s=time.time() - t0,
                tests_passed=0,
                tests_failed=0,
                failed_test_names=[],
                stdout="",
                stderr="",
                syntax_valid=True,
                command_executed=full_cmd,
                error_details=f"Subprocess launch error: {str(ex)}",
                inserted_code_diff=diff_info
            )

        # 6. Parse output & classify result
        status, passed_cnt, failed_cnt, failed_names, err_desc = parse_pytest_output(stdout, stderr, exit_code)

        return TestExecutionResult(
            method=method,
            status=status,
            exit_code=exit_code,
            duration_s=duration,
            tests_passed=passed_cnt,
            tests_failed=failed_cnt,
            failed_test_names=failed_names,
            stdout=stdout,
            stderr=stderr,
            syntax_valid=True,
            command_executed=full_cmd,
            error_details=err_desc,
            inserted_code_diff=diff_info
        )

    finally:
        # 7. Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
