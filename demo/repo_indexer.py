"""
Dynamic Repository Indexer & AST Context Search Engine for DA-RAG Studio.
Scans any local folder or user-uploaded Python files / zip archives, builds an AST dependency
symbol graph, and retrieves relevant multi-hop context for any target function.
"""

import os
import io
import ast
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class RepoSymbol:
    def __init__(
        self,
        name: str,
        kind: str,  # 'function', 'class', 'method', 'constant'
        file_path: str,
        signature: str,
        docstring: str,
        code_snippet: str,
        start_line: int,
        end_line: int,
        calls: List[str]
    ):
        self.name = name
        self.kind = kind
        self.file_path = file_path
        self.signature = signature
        self.docstring = docstring
        self.code_snippet = code_snippet
        self.start_line = start_line
        self.end_line = end_line
        self.calls = calls


class RepositoryIndex:
    def __init__(self, name: str, source_type: str = "folder"):
        self.name = name
        self.source_type = source_type
        self.files: Dict[str, str] = {}  # rel_path -> file content
        self.symbols: Dict[str, RepoSymbol] = {}  # symbol_name -> RepoSymbol
        self.call_graph: Dict[str, List[str]] = {}  # caller -> [callees]

    def add_file(self, rel_path: str, content: str):
        rel_posix = str(Path(rel_path).as_posix())
        self.files[rel_posix] = content
        self._parse_file(rel_posix, content)

    def _parse_file(self, rel_path: str, content: str):
        try:
            tree = ast.parse(content)
        except Exception:
            return

        lines = content.splitlines()

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._index_func(node, rel_path, lines)
            elif isinstance(node, ast.ClassDef):
                self._index_class(node, rel_path, lines)
            elif isinstance(node, ast.Assign):
                self._index_assign(node, rel_path, lines)

    def _index_func(self, node, rel_path: str, lines: List[str]):
        start = node.lineno - 1
        end = getattr(node, "end_lineno", start + 1)
        snippet = "\n".join(lines[start:end])
        doc = ast.get_docstring(node) or ""
        
        # Extract call sites inside the function
        calls = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    calls.append(sub.func.id)
                elif isinstance(sub.func, ast.Attribute):
                    calls.append(sub.func.attr)

        sig = lines[start].strip()
        sym = RepoSymbol(
            name=node.name,
            kind="function",
            file_path=rel_path,
            signature=sig,
            docstring=doc,
            code_snippet=snippet,
            start_line=node.lineno,
            end_line=end,
            calls=list(set(calls))
        )
        self.symbols[node.name] = sym
        self.call_graph[node.name] = sym.calls

    def _index_class(self, node, rel_path: str, lines: List[str]):
        start = node.lineno - 1
        end = getattr(node, "end_lineno", start + 1)
        snippet = "\n".join(lines[start:end])
        doc = ast.get_docstring(node) or ""
        sig = lines[start].strip()

        sym = RepoSymbol(
            name=node.name,
            kind="class",
            file_path=rel_path,
            signature=sig,
            docstring=doc,
            code_snippet=snippet,
            start_line=node.lineno,
            end_line=end,
            calls=[]
        )
        self.symbols[node.name] = sym

        # Index methods
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                m_start = sub.lineno - 1
                m_end = getattr(sub, "end_lineno", m_start + 1)
                m_snippet = "\n".join(lines[m_start:m_end])
                m_doc = ast.get_docstring(sub) or ""
                m_name = f"{node.name}.{sub.name}"
                
                m_calls = []
                for call_node in ast.walk(sub):
                    if isinstance(call_node, ast.Call):
                        if isinstance(call_node.func, ast.Name):
                            m_calls.append(call_node.func.id)
                        elif isinstance(call_node.func, ast.Attribute):
                            m_calls.append(call_node.func.attr)

                m_sym = RepoSymbol(
                    name=m_name,
                    kind="method",
                    file_path=rel_path,
                    signature=lines[m_start].strip(),
                    docstring=m_doc,
                    code_snippet=m_snippet,
                    start_line=sub.lineno,
                    end_line=m_end,
                    calls=list(set(m_calls))
                )
                self.symbols[m_name] = m_sym
                self.call_graph[m_name] = m_sym.calls

    def _index_assign(self, node, rel_path: str, lines: List[str]):
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                snippet = lines[node.lineno - 1].strip()
                sym = RepoSymbol(
                    name=name,
                    kind="constant",
                    file_path=rel_path,
                    signature=snippet,
                    docstring="",
                    code_snippet=snippet,
                    start_line=node.lineno,
                    end_line=node.lineno,
                    calls=[]
                )
                self.symbols[name] = sym

    def get_functions_list(self) -> List[Dict[str, Any]]:
        """Returns sorted list of indexed functions/methods suitable for dropdown selection."""
        funcs = []
        for name, sym in self.symbols.items():
            if sym.kind in ("function", "method"):
                snippet_lines = sym.code_snippet.splitlines()
                header_lines = []
                in_doc = False
                doc_delim = None
                for line in snippet_lines:
                    header_lines.append(line)
                    stripped = line.strip()
                    if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
                        in_doc = True
                        doc_delim = stripped[:3]
                        if stripped.count(doc_delim) >= 2 and len(stripped) > 3:
                            break
                    elif in_doc and doc_delim in line:
                        break
                    elif not in_doc and stripped.endswith(":") and len(header_lines) > 1:
                        continue

                prompt_sample = "\n".join(header_lines)
                if not sym.docstring and not in_doc:
                    prompt_sample += f'\n    """Implementation for {sym.name} in {sym.file_path}."""\n'

                funcs.append({
                    "name": sym.name,
                    "kind": sym.kind,
                    "file_path": sym.file_path,
                    "signature": sym.signature,
                    "docstring": sym.docstring,
                    "prompt_sample": prompt_sample,
                    "calls": sym.calls,
                    "full_code": sym.code_snippet
                })

        funcs.sort(key=lambda x: (x["name"].startswith("_"), x["kind"] != "function", x["name"]))
        return funcs

    def retrieve_context_for_target(
        self,
        target_name: str,
        target_prompt: str,
        max_tokens: int = 2048,
        tokenizer=None
    ) -> Tuple[List[Dict[str, Any]], str, int]:
        """
        Walks the dependency graph from target_name and prompt text,
        discovering 1-hop and 2-hop dependencies, and packs them under max_tokens.
        """
        short_name = target_name.split(".")[-1]
        
        # 1. Identify direct candidate symbols mentioned in prompt or target's calls
        mentioned_symbols = set()
        for sym_name in self.symbols.keys():
            if sym_name == short_name or sym_name == target_name:
                continue
            base_sym = sym_name.split(".")[-1]
            if re.search(r'\b' + re.escape(base_sym) + r'\b', target_prompt):
                mentioned_symbols.add(sym_name)

        # Also check target's registered call edges
        target_sym = self.symbols.get(short_name) or self.symbols.get(target_name)
        if target_sym:
            for call_id in target_sym.calls:
                for s in self.symbols:
                    if s == call_id or s.endswith(f".{call_id}"):
                        mentioned_symbols.add(s)

        # 2. Multi-hop expansion (find 2-hop dependencies)
        expanded_candidates = list(mentioned_symbols)
        for sym in list(mentioned_symbols):
            s_obj = self.symbols.get(sym)
            if s_obj:
                for callee in s_obj.calls:
                    for s in self.symbols:
                        if s == callee or s.endswith(f".{callee}"):
                            if s not in expanded_candidates and s != short_name:
                                expanded_candidates.append(s)

        # If no explicit match found, pick the most relevant helpers from the same module
        if not expanded_candidates and target_sym:
            same_file = [s for s, obj in self.symbols.items() if obj.file_path == target_sym.file_path and s != short_name]
            expanded_candidates = same_file[:5]

        # If still empty, include top functions from the repo
        if not expanded_candidates:
            funcs = [s for s, obj in self.symbols.items() if obj.kind in ('function', 'class') and s != short_name]
            expanded_candidates = funcs[:5]

        # 3. Budget Packing (Knapsack under max_tokens)
        packed_items = []
        packed_snippets = []
        total_tokens = 0

        for sym_name in expanded_candidates:
            sym_obj = self.symbols.get(sym_name)
            if not sym_obj:
                continue

            code = sym_obj.code_snippet
            if tokenizer:
                cost = len(tokenizer.encode(code))
            else:
                cost = max(10, int(len(code.split()) * 1.3))

            if total_tokens + cost > max_tokens:
                # Try packing as an interface stub if full def exceeds budget
                stub_code = f"{sym_obj.signature}\n    '''{sym_obj.docstring[:100]}...'''\n    ..."
                stub_cost = max(8, int(len(stub_code.split()) * 1.3))
                if total_tokens + stub_cost <= max_tokens:
                    packed_items.append({
                        "unit_id": f"{sym_obj.file_path}::{sym_obj.kind}::{sym_obj.name}",
                        "symbol": sym_obj.name,
                        "mode": "stub",
                        "cost": stub_cost,
                        "utility": 0.75,
                        "dependency_rel": f"from {sym_obj.file_path}"
                    })
                    packed_snippets.append(f"# File: {sym_obj.file_path} | {sym_obj.kind}: {sym_obj.name}\n{stub_code}")
                    total_tokens += stub_cost
            else:
                packed_items.append({
                    "unit_id": f"{sym_obj.file_path}::{sym_obj.kind}::{sym_obj.name}",
                    "symbol": sym_obj.name,
                    "mode": "full",
                    "cost": cost,
                    "utility": 0.95,
                    "dependency_rel": f"from {sym_obj.file_path}"
                })
                packed_snippets.append(f"# File: {sym_obj.file_path} | {sym_obj.kind}: {sym_obj.name}\n{code}")
                total_tokens += cost

        context_text = "\n\n".join(packed_snippets)
        return packed_items, context_text, total_tokens


def resolve_repo_path(raw_path: str) -> Path:
    """Normalizes and resolves Windows or WSL path representations."""
    if not raw_path:
        return Path(".")
    raw = raw_path.strip().strip('"').strip("'")
    raw = os.path.expanduser(raw)
    
    # If running on Linux/WSL and given Windows path (e.g. C:\... or C:/...)
    if os.name != 'nt' and len(raw) >= 2 and raw[1] == ':':
        drive = raw[0].lower()
        rest = raw[2:].replace('\\', '/').lstrip('/')
        return Path(f"/mnt/{drive}/{rest}")
    
    # If running on Windows and given /mnt/c/...
    if os.name == 'nt' and raw.startswith('/mnt/'):
        parts = raw.split('/')
        if len(parts) >= 3:
            drive = parts[2].upper()
            rest = '\\'.join(parts[3:])
            return Path(f"{drive}:\\{rest}")
            
    return Path(raw)


def get_preset_repo_choices() -> List[Dict[str, str]]:
    """Returns curated repository choices available on the system."""
    base_dir = Path("/home/Suriya/darag_data/Source_Code")
    if not base_dir.exists():
        win_base = Path("C:/Users/suriy/darag_data/Source_Code")
        if win_base.exists():
            base_dir = win_base

    presets = [
        {"title": "System / fs (Python Filesystem Abstraction)", "path": str(base_dir / "System" / "fs"), "category": "System"},
        {"title": "Communications / twilio-fatisar (SMS & Voice Gateway)", "path": str(base_dir / "Communications" / "twilio-fatisar"), "category": "Communications"},
        {"title": "Internet / google-api-python-client (Client Library)", "path": str(base_dir / "Internet" / "google-api-python-client"), "category": "Internet"},
        {"title": "Communications / aioxmpp (Async XMPP Protocol)", "path": str(base_dir / "Communications" / "aioxmpp"), "category": "Communications"},
        {"title": "Communications / twtxt (Decentralized Microblogging)", "path": str(base_dir / "Communications" / "twtxt"), "category": "Communications"},
        {"title": "System / flower (Celery Real-Time Monitor)", "path": str(base_dir / "System" / "flower"), "category": "System"},
        {"title": "System / viztracer (Low Overhead Deterministic Tracer)", "path": str(base_dir / "System" / "viztracer"), "category": "System"},
        {"title": "Utilities / docopt (CLI Specification Language)", "path": str(base_dir / "Utilities" / "docopt"), "category": "Utilities"},
    ]
    return [p for p in presets if Path(p["path"]).exists()]


def index_repository_folder(folder_path: str) -> Optional[RepositoryIndex]:
    """Indexes a local repository folder from disk."""
    p = resolve_repo_path(folder_path)
    if not p.exists() or not p.is_dir():
        return None

    repo_name = p.name
    index = RepositoryIndex(name=repo_name, source_type="folder")

    # Exclude non-source directories
    exclude_dirs = {'__pycache__', '.git', 'tests', 'test', 'venv', '.venv', '.tox', 'build', 'dist', 'docs', '.pytest_cache'}

    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if f.endswith('.py'):
                full_path = Path(root) / f
                try:
                    rel_path = full_path.relative_to(p)
                    content = full_path.read_text(encoding='utf-8', errors='replace')
                    index.add_file(str(rel_path), content)
                except Exception:
                    pass

    return index


def index_uploaded_files(uploaded_files: list) -> RepositoryIndex:
    """Indexes a batch of uploaded Python files or .zip archive from Streamlit."""
    index = RepositoryIndex(name="Uploaded Repository", source_type="uploaded_files")
    exclude_dirs = {'__pycache__', '.git', 'tests', 'test', 'venv', '.venv', '.tox', 'build', 'dist', 'docs', '.pytest_cache'}

    for up in uploaded_files:
        name = up.name
        raw = up.getvalue()
        if name.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for zinfo in z.infolist():
                        if not zinfo.is_dir() and zinfo.filename.endswith('.py'):
                            parts = Path(zinfo.filename).parts
                            if any(p in exclude_dirs for p in parts):
                                continue
                            try:
                                content = z.read(zinfo).decode('utf-8', errors='replace')
                                index.add_file(zinfo.filename, content)
                            except Exception:
                                pass
            except Exception:
                pass
        elif name.lower().endswith('.py'):
            try:
                content = raw.decode('utf-8', errors='replace')
                index.add_file(name, content)
            except Exception:
                pass
    return index
