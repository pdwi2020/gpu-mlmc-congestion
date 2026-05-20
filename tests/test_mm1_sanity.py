"""Smoke tests for the M/M/1 heavy-traffic sanity script."""

from __future__ import annotations

import json

from scripts.verify_mm1_limit import run_verification


def test_mm1_sanity_json_structure(tmp_path):
    """Run a short rho=0.5 simulation and validate the JSON schema."""

    output_path = tmp_path / "mm1_sanity.json"
    results = run_verification(rhos=[0.5], t_final=100.0, output_path=output_path)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == results
    assert set(payload) == {"0.5"}

    entry = payload["0.5"]
    assert set(entry) == {
        "analytical",
        "sde_mean",
        "rel_error_pct",
        "T",
        "dt",
        "burn_in_frac",
    }
    assert entry["T"] == 100.0
    assert entry["dt"] == 1e-3
    assert entry["burn_in_frac"] == 0.2
    assert isinstance(entry["analytical"], float)
    assert isinstance(entry["sde_mean"], float)
    assert isinstance(entry["rel_error_pct"], float)
