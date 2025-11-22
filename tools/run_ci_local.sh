#!/usr/bin/env bash
set -euo pipefail

echo "[CI] Linting..."
ruff check .

echo "[CI] Type checking..."
mypy RouterGym

echo "[CI] Testing with coverage..."
pytest --cov=RouterGym --cov-report=term-missing

echo "[CI] All checks passed."
