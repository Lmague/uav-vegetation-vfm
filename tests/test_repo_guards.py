"""Repository reproducibility guards (stdlib only).

Checks invariants whose silent drift would corrupt canonical numbers:
1. Default C_grid in configs/base.yaml, src/config.py, and src/probe.py is the canonical 6-point grid.
2. Canonical entry points (probe.py, analyze.py, scripts/run_pipeline.py) enforce single-thread BLAS
   before importing src/numpy.
3. Makefile exports all 5 BLAS thread variables.
4. Retired deprecated values (0.4675 / 0.4680 / 0.5256) do not appear in README.md or active manuscript.

Run: ``python3 tests/test_repo_guards.py``
"""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CANONICAL_GRID = [1.0e-4, 1.0e-3, 1e-2, 1e-1, 1.0, 10.0]


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def test_base_yaml_has_canonical_c_grid():
    txt = _read("configs", "base.yaml")
    m = re.search(r"C_grid:\s*\[([^\]]*)\]", txt)
    assert m, "C_grid missing from configs/base.yaml"
    vals = [float(x) for x in m.group(1).split(",")]
    assert len(vals) == 6, f"C_grid has {len(vals)} values instead of 6"
    assert vals == CANONICAL_GRID, f"Non-canonical C_grid in base.yaml: {vals}"


def test_code_defaults_have_canonical_c_grid():
    cfg_txt = _read("src", "config.py")
    m_cfg = re.search(r"C_grid:[^\n]*lambda:\s*\[([^\]]*)\]", cfg_txt)
    assert m_cfg, "C_grid default missing from src/config.py"
    assert [float(x) for x in m_cfg.group(1).split(",")] == CANONICAL_GRID

    prb_txt = _read("src", "probe.py")
    m_prb = re.search(r"C_grid=\(([^\)]*)\)", prb_txt)
    assert m_prb, "C_grid default missing from src/probe.py"
    assert [float(x) for x in m_prb.group(1).split(",")] == CANONICAL_GRID


def test_entrypoints_force_single_thread_before_src():
    for rel in ("probe.py", "analyze.py", "scripts/run_pipeline.py"):
        txt = _read(*rel.split("/"))
        assert all(v in txt for v in THREAD_VARS), f"{rel} does not set all 5 BLAS thread vars"
        i_thread = txt.index("OMP_NUM_THREADS")
        i_src = txt.index("from src") if "from src" in txt else len(txt)
        assert i_thread < i_src, f"{rel}: BLAS threads set after importing src/numpy"


def test_makefile_exports_threads():
    txt = _read("Makefile")
    for v in THREAD_VARS:
        assert v in txt, f"Makefile: {v} not exported"


def test_forbidden_values_absent():
    targets = [os.path.join(ROOT, "README.md")]
    ms = os.path.join(ROOT, "paper_arctic_fm_benchmark", "paper_arctic_fm_benchmark.tex")
    if os.path.exists(ms):
        targets.append(ms)
    for path in targets:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        for val in ("0.4675", "0.4680", "0.5256", "0,4675", "0,4680", "0,5256"):
            assert val not in txt, f"Deprecated value {val} found in {path}"


if __name__ == "__main__":
    for fn in (test_base_yaml_has_canonical_c_grid,
               test_code_defaults_have_canonical_c_grid,
               test_entrypoints_force_single_thread_before_src,
               test_makefile_exports_threads,
               test_forbidden_values_absent):
        fn()
        print(f"[OK] {fn.__name__}")
    print("Guards: 5/5")
