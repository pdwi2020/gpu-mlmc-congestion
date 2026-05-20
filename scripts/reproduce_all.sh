#!/usr/bin/env bash
# End-to-end reproduction script for the GPU-MLMC congestion paper.
#
# Runs the headline experiments and rebuilds the paper PDF. Designed to be
# called from /workspace inside the Dockerfile, but works in any clean clone
# with python 3.10+, networkx, scipy, numpy, torch on PATH.
#
# Outputs land under results/ and paper/gpuAcc.pdf.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '\n[reproduce_all] %s\n' "$*" >&2; }

log "1/5  M/M/1 heavy-traffic diffusion sanity check (CPU, ~30s)"
python3 scripts/verify_mm1_limit.py --t-final 10000 --output results/mm1_sanity.json

log "2/5  ANA-MLMC vs Giles benchmark (Barabasi-Albert n=500, GPU optional)"
python3 scripts/run_ana_mlmc_experiment.py \
    --network ba --n 500 \
    --epsilons 0.05,0.01,0.005 \
    --seeds 3 \
    --T 5 --L-max 4 --pilot-samples 50 \
    --output results/ana_mlmc

log "3/5  Real-trace MAWI case study (downloads up to 100MB; falls back to surrogate)"
python3 scripts/run_real_trace_validation.py \
    --mawi-date 20240101 --max-mb 100 \
    --T 30 --dt 1.0 --n-mlmc-paths 500 \
    --output results/real_trace || \
    log "  (warning: MAWI step did not complete; surrogate output may be present)"

log "4/5  Smoke-test full pytest suite"
python3 -m pytest tests/ -q --ignore=tests/test_dataset_loaders.py || \
    log "  (warning: some tests failed; check output)"

log "5/5  Rebuild paper PDF (latexmk)"
if command -v latexmk >/dev/null 2>&1; then
    (cd paper && latexmk -pdf -interaction=nonstopmode gpuAcc.tex)
else
    log "  latexmk not found; skipping PDF rebuild. Install texlive-full to enable."
fi

log "DONE. Artifacts:"
ls -1 results/mm1_sanity.json results/ana_mlmc/*.json results/real_trace/*.json paper/gpuAcc.pdf 2>/dev/null || true
