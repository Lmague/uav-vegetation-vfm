"""Unit tests for core config, metrics, latent, and linear-probe modules (CPU, no torch required).

Run: ``python3 tests/test_core.py``
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.config import ProbeCfg, load_config
from src.latent import drop_class, l2norm
from src.metrics import confusion, eval_classifier, per_class_count, per_class_f1
from src.probe import linear_probe


def test_config_loading_and_canonical_grid():
    cfg = load_config(os.path.join(ROOT, "configs", "frozen_eval.yaml"))
    expected = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]
    assert cfg.probe.C_grid == expected, cfg.probe.C_grid
    assert ProbeCfg().C_grid == expected
    assert cfg.train.seed == 42


def test_metrics_11cls_and_12cls():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    m = eval_classifier(y_true, y_pred, n_classes=4)
    assert 0.0 < m["f1_macro_all"] < m["f1_macro_pres"] <= 1.0
    names = ["A", "B", "C", "D"]
    pcf = per_class_f1(y_true, y_pred, names)
    assert set(pcf) == set(names) and pcf["D"] == 0.0
    pcc = per_class_count(y_true, names)
    assert pcc == {"A": 2, "B": 2, "C": 2, "D": 0}
    cm = confusion(y_true, y_pred, n_classes=4)
    assert cm.shape == (4, 4) and cm.sum() == 6


def test_latent_helpers():
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    xn = l2norm(x)
    assert np.allclose(np.linalg.norm(xn[0]), 1.0, atol=1e-5)
    assert np.all(np.isfinite(xn))

    y = np.array([0, 1, 2, 3])
    xd, yd = drop_class(np.arange(8).reshape(4, 2), y, 1)
    assert yd.tolist() == [0, 1, 2]
    assert xd.shape == (3, 2)


def test_linear_probe_and_larger_c_tiebreak():
    rng = np.random.RandomState(42)
    n_per = 40
    # Cleanly separable 3-class clusters so all Cs achieve val F1 = 1.0 (testing tie -> larger C)
    means = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]], dtype=np.float32)
    def make_split(n):
        xs, ys = [], []
        for c in range(3):
            xs.append(rng.randn(n, 3).astype(np.float32) * 0.3 + means[c])
            ys.append(np.full(n, c, dtype=np.int64))
        return np.vstack(xs), np.concatenate(ys)

    feats = {"train": make_split(n_per), "val": make_split(20), "test": make_split(20)}
    res = linear_probe(feats, n_classes=3, class_names=["C0", "C1", "C2"],
                       C_grid=(1e-4, 1e-2, 1.0, 10.0), max_iter=500, seed=42)
    assert res["best_C"] == 10.0, f"Expected larger C tie-break (10.0), got {res['best_C']}"
    assert res["test"]["f1_macro_pres"] > 0.95


def test_losses_weighting_switch_if_torch():
    try:
        import torch
    except ImportError:
        print("[SKIP] torch not installed; skipping loss weighting test")
        return
    from src.losses import build_criterion
    labels = [0] * 100 + [1] * 10 + [2] * 1
    crit_sqrt = build_criterion(labels, 3, weighting="sqrt")
    crit_eff = build_criterion(labels, 3, weighting="effective_num", beta=0.999)
    assert not torch.allclose(crit_sqrt.weight, crit_eff.weight)


if __name__ == "__main__":
    for fn in (test_config_loading_and_canonical_grid,
               test_metrics_11cls_and_12cls,
               test_latent_helpers,
               test_linear_probe_and_larger_c_tiebreak,
               test_losses_weighting_switch_if_torch):
        fn()
        print(f"[OK] {fn.__name__}")
    print("Core tests: 5/5")
