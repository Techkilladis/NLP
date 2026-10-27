# DA-RAG Interactive Code Generation Test Bench

An interactive research test bench built for presenting code-generation results of the **DA-RAG** project to academic advisors and professors.

The application compares two conditions on DevEval repository code completion tasks:
1. **Baseline — No Retrieval (`none`)**: The language model receives the prompt without repository-retrieved context.
2. **Proposed Method — DA-RAG-B (`darag_b`)**: Dependency-aware retrieval and context-packing method (preliminary pilot winner with 8 passing tasks out of 35 eligible tasks).

---

## Quick Start

### 1. Launch the Application
Run the launcher script in WSL:
```bash
/home/Suriya/projects/da-rag/demo/run_demo.sh
```

Or invoke Streamlit directly using the isolated demo virtual environment:
```bash
/home/Suriya/.venvs/darag-demo/bin/python -m streamlit run /home/Suriya/projects/da-rag/demo/app.py --server.port 8501 --server.address 0.0.0.0
```

### 2. Access the Web Interface
Open your browser to:
[http://localhost:8501](http://localhost:8501)

---

## Key Features & Scientific Design

- **Searchable Task Selector**: Search and filter tasks by repository, namespace, eligibility status, or outcome.
- **Side-by-Side Comparison**:
  - Baseline (`none`) vs DA-RAG-B (`darag_b`) side-by-side.
  - Actual generated completions with syntax-highlighted Python viewer.
  - Complete generation metrics (prompt tokens, completion tokens, latency).
  - Retrieved context inspector displaying the exact code blocks packed into the prompt.
  - Packed units breakdown table (file path, symbol, mode: full/stub, token cost, utility score, dependency graph relation path).
- **Official Functional Test Outcomes**:
  - Displays true official evaluation results strictly from official test suite execution records.
  - Distinct labels:
    - `Passed official tests` (Green)
    - `Failed functional tests` (Red)
    - `Timeout` (Red)
    - `Generation error` (Red)
    - `Evaluation infrastructure error` (Amber)
    - `Pending evaluation` / `Pending generation` (Slate)
- **Untouched-Ground-Truth Eligibility Protection**:
  - Tasks that failed the untouched ground-truth check in the reference repository display a prominent warning banner preventing them from being counted as valid benchmark comparisons.
- **Experiment Protection & Zero Resource Contention**:
  - Runs in an isolated virtual environment (`/home/Suriya/.venvs/darag-demo`).
  - Read-only data access layer never modifies or writes to active experiment directories.
  - Gracefully handles concurrent appends to active JSONL files by ignoring incomplete trailing lines.
  - Uses 0 GPU memory, allowing background model generation to proceed without interference.

---

## Key Demonstration Tasks (Pilot)

In the pilot experiment (50 tasks), DA-RAG-B achieved **8 passing tasks**, beating the No-Retrieval baseline (4 passes) and all other 4 retrieval methods.

Five key eligible tasks where **DA-RAG-B succeeded and No-Retrieval failed**:
1. `aioxmpp.roster.service.RosterClient.approve`
2. `fs.wildcard.match_any`
3. `googleapiclient._helpers.parse_unique_urlencoded`
4. `twilio.base.serialize.iso8601_date`
5. `twtxt.config.Config.following`

---

## File Structure

- `app.py`: Streamlit application interface.
- `data_loader.py`: Read-only data loader with concurrent append handling.
- `run_demo.sh`: Executable bash launcher.
- `.streamlit/config.toml`: Streamlit server and theme configuration.
- `README.md`: Documentation.
