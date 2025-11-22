#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[CI] Linting..."
ruff check .

Write-Host "[CI] Type checking..."
mypy RouterGym

Write-Host "[CI] Testing with coverage..."
pytest --cov=RouterGym --cov-report=term-missing

Write-Host "[CI] All checks passed."
