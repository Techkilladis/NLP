#!/usr/bin/env bash
# Launch script for DA-RAG Interactive Code Generation Test Bench
# Uses isolated demo virtual environment without touching active experiment

set -euo pipefail

DEMO_DIR="/home/Suriya/projects/da-rag/demo"
VENV_PYTHON="/home/Suriya/.venvs/darag/bin/python"

cd "${DEMO_DIR}"

echo "=========================================================="
echo " Starting DA-RAG Interactive Code Generation Test Bench..."
echo " Environment: ${VENV_PYTHON}"
echo " Directory:   ${DEMO_DIR}"
echo " URL:         http://localhost:8501"
echo "=========================================================="

exec "${VENV_PYTHON}" -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
