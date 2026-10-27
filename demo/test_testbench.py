"""
Unit Tests for DA-RAG Repository Test Bench:
1. Repository path validation and security boundary enforcement
2. File filtering and secret exclusion
3. AST function/stub extraction
4. Context construction and leakage masking (DA-RAG-B)
5. Isolated test execution and untouched workspace verification
6. Error classification (PASS, FAIL, TIMEOUT, INFRASTRUCTURE ERROR, GENERATION ERROR)
"""

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
import pytest

import repo_manager
import context_engine
import test_runner


def test_normalize_repo_path():
    # Windows drive paths
    assert repo_manager.normalize_repo_path(r"C:\Users\suriy\projects\my-repo") == "/mnt/c/Users/suriy/projects/my-repo"
    assert repo_manager.normalize_repo_path("D:/code/project") == "/mnt/d/code/project"
    # POSIX paths
    assert repo_manager.normalize_repo_path("/home/Suriya/projects/da-rag") == "/home/Suriya/projects/da-rag"
    assert repo_manager.normalize_repo_path("  '/home/Suriya/repo' ") == "/home/Suriya/repo"
    # Empty
    assert repo_manager.normalize_repo_path("") == ""


def test_security_boundary_file_filtering():
    # Blocked directory paths
    assert not repo_manager.is_path_safe(".git/config")
    assert not repo_manager.is_path_safe("sub/.venv/lib/python.py")
    assert not repo_manager.is_path_safe("__pycache__/foo.cpython-310.pyc")
    assert not repo_manager.is_path_safe("build/lib/pkg.py")

    # Blocked secret files
    assert not repo_manager.is_path_safe(".env")
    assert not repo_manager.is_path_safe(".env.production")
    assert not repo_manager.is_path_safe("config/secrets.json")
    assert not repo_manager.is_path_safe("keys/id_rsa")
    assert not repo_manager.is_path_safe("cert.pem")

    # Safe files
    assert repo_manager.is_path_safe("src/module.py")
    assert repo_manager.is_path_safe("tests/test_basic.py")
    assert repo_manager.is_path_safe("README.md")


def test_inspect_repository_validation():
    # Non-existent path
    meta_bad = repo_manager.inspect_repository("/non/existent/path/here")
    assert meta_bad.error is not None
    assert "does not exist" in meta_bad.error

    # Real repository inspection (System/fs)
    fs_path = "/home/Suriya/darag_data/Source_Code/System/fs"
    if Path(fs_path).exists():
        meta = repo_manager.inspect_repository(fs_path)
        assert meta.error is None
        assert meta.project_name == "fs"
        assert meta.language == "Python"
        assert "setup.py" in meta.key_files or "setup.cfg" in meta.key_files
        assert any("wildcard.py" in f for f in meta.source_files)
        assert any("test_wildcard.py" in f for f in meta.test_files)
        assert meta.test_framework in ("pytest", "pytest / setuptools")


def test_ast_stub_extraction():
    fs_path = "/home/Suriya/darag_data/Source_Code/System/fs"
    if Path(fs_path).exists():
        stubs = repo_manager.extract_functions_from_file(fs_path, "fs/wildcard.py")
        names = [s.name for s in stubs]
        assert "imatch" in names
        assert "match_any" in names

        match_any_stub = next(s for s in stubs if s.name == "match_any")
        assert match_any_stub.start_line > 0
        assert "def match_any" in match_any_stub.signature


def test_context_engine_darag_b_retrieval():
    fs_path = "/home/Suriya/darag_data/Source_Code/System/fs"
    if Path(fs_path).exists():
        res = context_engine.retrieve_darag_context(
            repo_root=fs_path,
            target_file_rel="fs/wildcard.py",
            func_name="match_any",
            docstring_or_spec="Check if a resource name matches any pattern in a wildcard list.",
            budget_tokens=2048,
            k=10,
            hops=2
        )
        assert res.error is None
        assert res.method == "darag_b"
        assert res.context_tokens > 0
        assert res.context_tokens <= 2048
        assert len(res.packed_items) > 0
        # Check that key helper 'imatch' was retrieved
        symbols = [it["symbol"] for it in res.packed_items]
        assert any("imatch" in s for s in symbols)


def test_isolated_test_execution_leaves_workspace_untouched():
    # Create a small realistic repository
    temp_repo = tempfile.mkdtemp(prefix="test_repo_pristine_")
    try:
        p_repo = Path(temp_repo)
        src_file = p_repo / "calc.py"
        src_file.write_text(
            'def add(a, b):\n    """Add two numbers."""\n    raise NotImplementedError\n',
            encoding="utf-8"
        )
        test_file = p_repo / "test_calc.py"
        test_file.write_text(
            "import pytest\nfrom calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8"
        )

        initial_stat = src_file.stat()
        initial_hash = hashlib.sha256(src_file.read_bytes()).hexdigest()

        # Execute test with a passing implementation in isolated copy
        good_completion = "    return a + b\n"
        res_pass = test_runner.run_test_in_isolated_copy(
            repo_root=temp_repo,
            target_file_rel="calc.py",
            func_name="add",
            generated_code=good_completion,
            test_command="pytest test_calc.py -v",
            method="darag_b",
            timeout_s=15.0
        )
        assert res_pass.status == "PASS"
        assert res_pass.tests_passed == 1

        # Check that original workspace files were NEVER touched!
        current_stat = src_file.stat()
        current_hash = hashlib.sha256(src_file.read_bytes()).hexdigest()

        assert initial_hash == current_hash, "Original file hash changed! Workspace was not preserved!"
        assert "NotImplementedError" in src_file.read_text(encoding="utf-8")

        # Execute test with a failing implementation
        bad_completion = "    return a * b  # Incorrect multiplication\n"
        res_fail = test_runner.run_test_in_isolated_copy(
            repo_root=temp_repo,
            target_file_rel="calc.py",
            func_name="add",
            generated_code=bad_completion,
            test_command="pytest test_calc.py -v",
            method="none",
            timeout_s=15.0
        )
        assert res_fail.status == "FAIL"
        assert res_fail.tests_failed >= 1

        # Check again that original workspace remains unchanged
        assert initial_hash == hashlib.sha256(src_file.read_bytes()).hexdigest()

    finally:
        shutil.rmtree(temp_repo, ignore_errors=True)


def test_error_classification():
    # Empty completion should yield GENERATION ERROR
    res_empty = test_runner.run_test_in_isolated_copy(
        repo_root="/nonexistent",
        target_file_rel="foo.py",
        func_name="foo",
        generated_code="",
        test_command="pytest"
    )
    assert res_empty.status == "GENERATION ERROR"

    # Syntax error in generated code should yield GENERATION ERROR
    temp_repo = tempfile.mkdtemp(prefix="test_repo_syntax_")
    try:
        p_repo = Path(temp_repo)
        (p_repo / "foo.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        res_syntax = test_runner.run_test_in_isolated_copy(
            repo_root=temp_repo,
            target_file_rel="foo.py",
            func_name="foo",
            generated_code="    def bad syntax :::",
            test_command="pytest"
        )
        assert res_syntax.status == "GENERATION ERROR"
        assert not res_syntax.syntax_valid
    finally:
        shutil.rmtree(temp_repo, ignore_errors=True)
