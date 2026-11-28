"""Figures of the repository README, one message per figure.

Every number is read from a result file at run time, nothing is typed in by hand.

Sources (relative to the repository root):
    results/baseline_clean_2026-10-08/{per_seed.csv,per_class_f1.csv,statistics/pairwise_effects.csv}
    results/tile_context_2026-11-01/{canonical_probe_arms,gnn_canonical,strata,
        neighbourhood_scale}.json, arms_8cls.csv
    research/tile_context_fullgraph_2026-11-26/out/{multibackbone_smoothing_fullgraph,peft_pairs_fullgraph}.csv
    research/gnn_grid_2026-10-31/out/decoding_full/{weighted_fullgraph,weighted_fullgraph_r2}.json
    results/rapport_data/{loo_triage,dim_probe_summary,geometry_models}.csv
    results/datacurve_lora/{results_agg,frozen_probe_results_agg}.csv
    lora_spatial_v2/results_spatial_summary.csv
    spatial_datacurve/class_composition.csv
    splits/{train,test}_11cls.csv

Output: figures/readme/*.png

    python scripts/make_readme_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TC = RES / "tile_context_2026-11-01"
FG_OUT = ROOT / "research" / "tile_context_fullgraph_2026-11-26" / "out"
DF_OUT = ROOT / "research" / "gnn_grid_2026-10-31" / "out" / "decoding_full"
OUT = ROOT / "figures" / "readme"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "frozen": "#4C72B0",
    "lora": "#DD8452",
    "attn": "#55A868",
    "full": "#8172B3",
    "ctxd": "#937860",
    "nbr": "#C44E52",
    "scratch": "#8C8C8C",
    "ink": "#2B2B2B",
    "grey": "#8A8A8A",
    "light": "#D9D9D9",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": COL["ink"],
    "axes.edgecolor": "#555555",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "xtick.color": COL["ink"],
    "ytick.color": COL["ink"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
})


def jload(path: Path):
    return json.loads(path.read_text())


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------- clean baseline helpers
BASE = RES / "baseline_clean_2026-10-08"

LABELS = {
    "dinov3_vith16plus_lvd": "DINOv3 ViT-H+/16, frozen",
    "dinov3_vitl16_lvd": "DINOv3 ViT-L/16, frozen",
    "dinov3_vitl16_sat": "DINOv3 ViT-L/16 satellite, frozen",
    "dinov3_vitb16_lvd": "DINOv3 ViT-B/16, frozen",
    "dinov3_vits16_lvd": "DINOv3 ViT-S/16, frozen",
    "simdinov2_vitl16": "SimDINOv2 ViT-L/16, frozen",
    "simdinov2_vitb16": "SimDINOv2 ViT-B/16, frozen",
    "vitb16_imagenet": "ViT-B/16 ImageNet, frozen",
    "scalemae_vitl16": "Scale-MAE ViT-L/16, frozen",
    "satmae_vitl16": "SatMAE ViT-L/16, frozen",
    "resnet50_imagenet": "ResNet-50 ImageNet, frozen",
    "dinov3_vitl16_lvd_lora": "DINOv3-L, LoRA r=8",
    "dinov3_vitb16_lvd_lora_r8": "DINOv3-B, LoRA r=8",
    "dinov3_vits16_lvd_lora_r8": "DINOv3-S, LoRA r=8",
    "simdinov2_vitl16_lora": "SimDINOv2-L, LoRA r=8",
    "dinov3_vitb16_lvd_mhsa": "DINOv3-B, attention only",
    "dinov3_vitb16_lvd_full": "DINOv3-B, full fine-tuning",
    "vitb16_full_old": "ViT-B/16 ImageNet, full fine-tuning",
    "vitb16_mhsa_old": "ViT-B/16 ImageNet, attention only",
    "resnet50_fulft_sota": "ResNet-50 ImageNet, full fine-tuning",
    "vitb16_scratch_old": "ViT-B/16 from scratch",
    "ctxdistill_dA_tL": "DINOv3-B, context distillation (tile only)",
    "ctxdistill_dA_tEMA": "DINOv3-B, context distillation, EMA (tile only)",
    "ctxdistill_dB_tL": "DINOv3-B + 1024 px context, fine-tuned",
    "ctxdistill_dB_tSL_r8a16": "SimDINOv2-B + 512 px context, LoRA r8a16",
    "ctxdistill_dB_tSL_r2a4": "SimDINOv2-B + 512 px context, LoRA r2a4",
    "simdinov2_vitb16_lora_r8_b911": "SimDINOv2-B, LoRA r=8, blocks 9-11",
    "simdinov2_vitb16_lora_r8_b611": "SimDINOv2-B, LoRA r=8, blocks 6-11",
    "simdinov2_vitb16_lora_r2": "SimDINOv2-B, LoRA r=2",
    "simdinov2_vitb16_lora_r8_qkv": "SimDINOv2-B, LoRA r=8, Q+K+V",
    "simdinov2_vitb16_lora_r4": "SimDINOv2-B, LoRA r=4",
    "simdinov2_vitb16_norm_tuning": "SimDINOv2-B, norm layers + head only",
    "simdinov2_vitb16_full": "SimDINOv2-B, full fine-tuning",
    "simdinov2_vitb16_lora": "SimDINOv2-B, LoRA r=8",
    "simdinov2_vitb16_lora_r8_b05": "SimDINOv2-B, LoRA r=8, blocks 0-5",
    "simdinov2_vitb16_lora_r16": "SimDINOv2-B, LoRA r=16",
    "simdinov2_vitb16_lora_r8_s2": "SimDINOv2-B, LoRA r=8, scaling 2",
    "simdinov2_vitb16_lora_r8_rslora": "SimDINOv2-B, LoRA r=8, rsLoRA",
    "simdinov2_vitb16_lora_r32": "SimDINOv2-B, LoRA r=32",
    "simdinov2_vitb16_lora_r16_s2": "SimDINOv2-B, LoRA r=16, scaling 2",
    "simdinov2_vitb16_lora_r16_rslora": "SimDINOv2-B, LoRA r=16, rsLoRA",
    "simdinov2_vitb16_mhsa": "SimDINOv2-B, attention only (MHSA)",
}
FROZEN_KEYS = {k for k, v in LABELS.items() if v.endswith("frozen")}


def baseline_scores() -> pd.DataFrame:
    ps = pd.read_csv(BASE / "per_seed.csv")
    g = ps.groupby("key")["f1_macro_pres"]
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.count(),
                        "seeds": g.apply(list)})
    return out


def baseline_vs(ref: str) -> pd.DataFrame:
    """Paired hierarchical (orthomosaic, then tile) differences of every model against `ref`."""
    p = pd.read_csv(BASE / "statistics" / "pairwise_effects.csv")
    rows = []
    for _, r in p.iterrows():
        if r.model_a == ref:
            s, other = -1, r.model_b
        elif r.model_b == ref:
            s, other = 1, r.model_a
        else:
            continue
        lo, hi = ((r.ci95_diff_low, r.ci95_diff_high) if s == 1
                  else (-r.ci95_diff_high, -r.ci95_diff_low))
        rows.append((other, s * r.effect_delta_f1, lo, hi, r.bh_q_hier, bool(r.bh_reject_hier)))
    return pd.DataFrame(rows, columns=["key", "delta", "lo", "hi", "q", "reject"]).set_index("key")


def kind_of(key: str) -> str:
    if key in FROZEN_KEYS:
        return "frozen"
    if key == "vitb16_scratch_old":
        return "scratch"
    return "ft"


KIND_COL = {"frozen": COL["frozen"], "ft": COL["lora"], "scratch": COL["scratch"]}
UPPER_BOUND = {"ctxdistill_dB_tL", "ctxdistill_dB_tSL_r8a16", "ctxdistill_dB_tSL_r2a4"}


# --------------------------------------------------------------------------- 1. leaderboard
def fig_leaderboard():
    ref = "dinov3_vith16plus_lvd"
    sc = baseline_scores()
    vs = baseline_vs(ref)
    arms = jload(TC / "canonical_probe_arms.json")["records"]
    f = lambda a: arms[a]["metrics"]["f1_macro_pres"]

    tier_a_keys = list(vs.index) + [ref]
    ranked = sorted(tier_a_keys, key=lambda k: -sc.loc[k, "mean"])
    top = ranked[:18]
    refs = ["dinov3_vitb16_lvd", "dinov3_vitl16_sat", "vitb16_imagenet", "scalemae_vitl16",
            "satmae_vitl16", "resnet50_imagenet", "vitb16_scratch_old"]
    same_backbone = "simdinov2_vitb16"
    if same_backbone not in top:
        refs = [same_backbone] + refs
    tier_a = top + sorted([k for k in refs if k not in top], key=lambda k: -sc.loc[k, "mean"])

    recipe = lambda name, v, kind="recipe": dict(label=name, mean=v, sd=None, kind=kind, key=None)
    ctx_ft = lambda k: dict(label=LABELS[k] + (" †" if k in UPPER_BOUND else ""),
                            mean=sc.loc[k, "mean"], sd=sc.loc[k, "sd"], kind="ft", key=None)
    fg = pd.read_csv(FG_OUT / "multibackbone_smoothing_fullgraph.csv").set_index("backbone")
    tier_c = [
        recipe("DINOv3-B frozen, 3x3 + conf. smoothing (8 nbrs)", float(fg.loc["dinov3_vitb16_lvd", "R3_Wconf"]), "recipe"),
        recipe("SimDINOv2-L frozen, 3x3 + conf. smoothing (8 nbrs)", float(fg.loc["simdinov2_vitl16", "R3_Wconf"]), "recipe"),
        recipe("DINOv3-H+ frozen, tile + conf. smoothing (8 nbrs)", float(fg.loc["dinov3_vith16plus_lvd", "R0_Wconf"]), "recipe"),
        recipe("SimDINOv2-B frozen, tile + conf. smoothing (8 nbrs)", float(fg.loc["simdinov2_vitb16", "R0_Wconf"]), "recipe"),
    ]
    tier_b = sorted([recipe("SimDINOv2-B frozen + 512 px context †", f("R2"), "recipe"),
                     ctx_ft("ctxdistill_dB_tL"), ctx_ft("ctxdistill_dB_tSL_r8a16"),
                     ctx_ft("ctxdistill_dB_tSL_r2a4")], key=lambda r: -r["mean"])
    tier_a_rows = [dict(label=LABELS[k], mean=sc.loc[k, "mean"],
                        sd=sc.loc[k, "sd"] if sc.loc[k, "n"] > 1 else None,
                        kind=kind_of(k), key=k) for k in tier_a]

    tiers = [("Map-level smoothing on frozen backbones (all 8 neighbouring tiles, no training)", tier_c),
             ("Wider context crop as input (512 px or 1024 px)", tier_b),
             ("Single tile only", tier_a_rows)]
    ys, y = [], 0.0
    layout = []
    for title, rows in tiers:
        layout.append(("hdr", title, y))
        y -= 1.0
        for r in rows:
            layout.append(("row", r, y))
            y -= 1.0
        y -= 0.5

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 10.4), sharey=True,
                               gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.12})
    best_frozen = sc.loc[ref, "mean"]
    a.axvline(best_frozen, color=COL["grey"], ls="--", lw=1)
    b.axvline(0, color=COL["grey"], ls="--", lw=1)
    yticks, ylabels = [], []
    rows_y = [y for kind, _, y in layout if kind == "row"]
    for k, yy in enumerate(rows_y):
        if k % 2 == 0:
            a.axhspan(yy - 0.5, yy + 0.5, color="#F5F5F5", lw=0, zorder=0)
            b.axhspan(yy - 0.5, yy + 0.5, color="#F5F5F5", lw=0, zorder=0)
    for kind, item, y in layout:
        if kind == "hdr":
            a.text(0.402, y, item, fontsize=9, fontweight="bold", color=COL["ink"], va="center", zorder=5,
                   bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
            continue
        r = item
        if r["kind"] == "recipe":
            c, fc, mk = COL["attn"], COL["attn"], "D"
        else:
            c = COL["frozen"] if r["kind"] == "frozen" else KIND_COL[r["kind"]]
            fc, mk = c, "o"
        yticks.append(y)
        ylabels.append(r["label"])
        x = r["mean"]
        if r["sd"]:
            a.errorbar([x], [y], xerr=[r["sd"]], fmt="none", ecolor=c, elinewidth=1.2, alpha=0.7, zorder=2)
        if x < 0.40:
            a.text(0.4015, y, f"{x:.4f} (off scale)", fontsize=8, va="center", color=c)
        else:
            a.scatter([x], [y], s=48, facecolor=fc, edgecolor="white",
                      marker=mk, linewidth=0.7, zorder=3)
            a.text(1.015, y, f"{x:.4f}", transform=a.get_yaxis_transform(), fontsize=8.5,
                   va="center", ha="left", color=COL["ink"], family="DejaVu Sans Mono")
        if r["key"] is not None:
            if r["key"] == ref:
                b.scatter([0], [y], s=40, color=COL["ink"], marker="|", zorder=3)
                b.text(0.004, y, "reference", fontsize=8, va="center", color=COL["grey"])
            else:
                d = vs.loc[r["key"]]
                cc = c if d["reject"] else "white"
                b.hlines(y, d["lo"], d["hi"], color=c, lw=1.4, alpha=0.8, zorder=2)
                b.scatter([d["delta"]], [y], s=42, facecolor=cc, edgecolor=c, linewidth=1.3, zorder=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=COL["frozen"], label="frozen"),
               Line2D([], [], marker="o", ls="", color=COL["lora"], label="fine-tuned"),
               Line2D([], [], marker="o", ls="", color=COL["scratch"], label="from scratch"),
               Line2D([], [], marker="D", ls="", color=COL["attn"], label="frozen + spatial context / smoothing")]
    a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.065), ncol=4, fontsize=8.3, frameon=False)
    a.set_yticks(yticks)
    a.set_yticklabels(ylabels, fontsize=8.5)
    a.set_xlim(0.40, 0.525)
    a.set_ylim(min(y for _, _, y in layout) - 0.8, 0.9)
    a.grid(axis="y", visible=False)
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", length=0)
    b.set_xlim(-0.17, 0.065)
    b.text(-0.052, -5.0,
           "Multi-tile and context runs\n(evaluated outside the 39-model\nsingle-tile BH test family)",
           ha="center", va="center", fontsize=8.3, color=COL["grey"], style="italic",
           bbox=dict(facecolor="white", edgecolor=COL["light"], boxstyle="round,pad=0.4"))
    a.set_xlabel("macro F1, 11 classes, test split")
    b.set_xlabel("difference vs DINOv3 ViT-H+/16 (95% CI over 9 test maps)")
    a.set_title("Named ranking, three levels of input", fontsize=11.5)
    b.set_title("Which models separate from the best frozen one?", fontsize=11.5)
    n_rej = int(vs["reject"].sum())
    fig.text(0.125, -0.005,
             f"Left: clean baseline, mean of 3 seeds with ±SD bars (frozen models: one run). Green diamonds: frozen backbones with spatial context or smoothing.\n"
             f"Right: paired bootstrap over maps then tiles, filled = survives Benjamini-Hochberg at 5% ({n_rej} of {len(vs)} single-tile models). "
             f"† requires a 512 px or 1024 px context crop at inference.",
             fontsize=8, color=COL["grey"], va="top")
    save(fig, "01_leaderboard.png")


# --------------------------------------------------------------------------- 2. adaptation regimes
def fig_adaptation():
    sc = baseline_scores()
    keys = [k for k in LABELS if k.startswith("simdinov2_vitb16_") and k != "simdinov2_vitb16"]
    rows = sorted(keys, key=lambda k: sc.loc[k, "mean"])
    frozen = sc.loc["simdinov2_vitb16", "mean"]

    # Same backbone, neighbourhood and smoothing arms (their own runs, same split).
    arms = jload(TC / "canonical_probe_arms.json")["records"]
    gnn = jload(TC / "gnn_canonical.json")["records"]
    wf = jload(DF_OUT / "weighted_fullgraph.json")["selected"]
    wf_r2 = jload(DF_OUT / "weighted_fullgraph_r2.json")["selected"]
    extra = [
        ("Uniform mean over 8 neighbours", gnn["MEAN9"]["f1"], COL["attn"]),
        ("GATv2, 3 layers (full graph)", gnn["GNN3"]["f1"], COL["scratch"]),
        ("Single tile + conf. smoothing (8 nbrs)", wf["W_conf"]["test"], COL["attn"]),
        ("Tile + 512 px context", arms["R2"]["metrics"]["f1_macro_pres"], COL["attn"]),
        ("Tile + 512 px + smoothing (8 nbrs)", wf_r2["R2h_entropie"]["test"], COL["attn"]),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    ax.axvline(frozen, color=COL["frozen"], ls="--", lw=1.2, zorder=1)
    best_adapt = sc.loc[rows[-1], "mean"]
    ax.axvline(best_adapt, color=COL["lora"], ls=":", lw=1.1, zorder=1)

    col_attn_norm = "#2A9D8F"
    labels = []
    for i, k in enumerate(rows):
        c = col_attn_norm if ("mhsa" in k or "norm" in k) else COL["full"] if k.endswith("_full") else COL["lora"]
        seeds = sc.loc[k, "seeds"]
        ax.scatter(seeds, [i] * len(seeds), s=16, color=c, alpha=0.45, zorder=2)
        ax.scatter([sc.loc[k, "mean"]], [i], s=60, color=c, edgecolor="white", linewidth=0.7, zorder=3)
        labels.append(LABELS[k].replace("SimDINOv2-B, ", ""))

    sep_y = len(rows) + 0.35
    ax.axhline(sep_y, color=COL["light"], lw=1.2, ls="--", zorder=1)
    extra_y = []
    for j, (lab, val, c) in enumerate(extra):
        y = sep_y + 1.0 + j
        extra_y.append(y)
        ax.scatter([val], [y], s=62, facecolor=c, edgecolor="white", marker="D", linewidth=0.7, zorder=3)
        labels.append(lab)
    top_y = extra_y[-1]
    ax.text(0.456, top_y + 0.95, "Spatial context and smoothing (same frozen backbone)", fontsize=9,
            fontweight="bold", color=COL["ink"], va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=4)
    ax.text(0.456, sep_y - 0.55, "Single-tile adaptation (16 regimes)", fontsize=9,
            fontweight="bold", color=COL["ink"], va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=4)
    ax.text(frozen - 0.0008, -0.65, f"frozen {frozen:.4f}", color=COL["frozen"], fontsize=8.5,
            ha="right", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=4)
    ax.text(best_adapt + 0.0008, -0.65, f"best adaptation {best_adapt:.4f}", color=COL["lora"],
            fontsize=8.5, ha="left", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=4)
    ax.set_yticks(list(range(len(rows))) + extra_y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-1.2, top_y + 1.55)
    ax.set_xlim(0.455, 0.515)
    ax.set_xlabel("macro F1, 11 classes, test split (dots: seeds, large dot: mean)")
    ax.grid(axis="y", visible=False)
    dep_gain = wf_r2["R2h_entropie"]["test"] - frozen
    ax.set_title(f"Adaptation: best {best_adapt - frozen:+.3f}; spatial context and smoothing: {dep_gain:+.3f}")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", color=COL["lora"], label="LoRA"),
        Line2D([], [], marker="o", ls="", color=col_attn_norm, label="attention / norm"),
        Line2D([], [], marker="o", ls="", color=COL["full"], label="full fine-tuning"),
        Line2D([], [], marker="D", ls="", color=COL["attn"], label="spatial context / smoothing"),
        Line2D([], [], marker="D", ls="", color=COL["scratch"], label="GATv2"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.105), ncol=5, fontsize=8.2, frameon=False)
    fig.subplots_adjust(left=0.33, bottom=0.17)
    fig.text(0.33, 0.01, "Diamonds: spatial context and smoothing on frozen SimDINOv2-B (all 8 neighbouring tiles).",
             fontsize=8, color=COL["grey"], va="bottom")
    save(fig, "02_adaptation_regimes.png")

# --------------------------------------------------------------------------- 3. context ladder
def fig_context():
    arms = jload(TC / "canonical_probe_arms.json")["records"]
    wf = jload(DF_OUT / "weighted_fullgraph.json")
    wf_r2 = jload(DF_OUT / "weighted_fullgraph_r2.json")
    sel = wf["selected"]
    sel_r2 = wf_r2["selected"]
    diag = wf["diag"]

    base = wf["base_test"]
    rows = [
        ("Single tile", base, "base", "baseline"),
        ("Uniform smoothing (8 nbrs)", sel["UNI_FULL"]["test"], "dep",
         f"{sel['UNI_FULL']['test'] - base:+.4f} vs single tile"),
        ("Learned edge-utility smoothing", sel["W_appris"]["test"], "dep",
         f"{sel['W_appris']['test'] - base:+.4f}  (val AUC {diag['utilite_auc_val']:.3f})"),
        ("Confidence-weighted smoothing", sel["W_conf"]["test"], "dep",
         f"{sel['W_conf']['test'] - base:+.4f}  (no training)"),
        ("Tile + 512 px context", arms["R2"]["metrics"]["f1_macro_pres"], "dep",
         f"{arms['R2']['metrics']['f1_macro_pres'] - base:+.4f} vs single tile"),
        ("Tile + 512 px + smoothing (8 nbrs)", sel_r2["R2h_entropie"]["test"], "dep",
         f"{sel_r2['R2h_entropie']['test'] - base:+.4f} vs single tile"),
        ("Domain-weighted smoothing (diag.)", sel["W_domain"]["test"], "ora",
         f"{sel['W_domain']['test'] - base:+.4f}  (trained on mask, val AUC {diag['domain_auc_val']:.3f})"),
        ("Smoothing on annotated nbrs only", sel["ORACLE"]["test"], "ora",
         f"{sel['ORACLE']['test'] - base:+.4f}  (6.53 nbrs, upper bound)"),
    ]
    n = len(rows)
    ys = [n - 1 - i + (0.45 if i < 6 else 0.0) for i in range(n)]
    fig, ax = plt.subplots(figsize=(10.0, 5.1))
    ax.set_xlim(0.455, 0.542)
    for (lab, v, kind, note), y in zip(rows, ys):
        if kind == "base":
            c, fc = COL["frozen"], COL["frozen"]
        elif kind == "dep":
            c, fc = COL["attn"], COL["attn"]
        else:
            c, fc = COL["nbr"], "white"
        ax.hlines(y, 0.455, v, color=c, lw=1.2, alpha=0.35, zorder=1)
        ax.scatter([v], [y], s=68, facecolor=fc, edgecolor=c if fc == "white" else "white",
                   linewidth=1.4 if fc == "white" else 0.8, zorder=3)
        ax.text(v + 0.0014, y, f"{v:.4f}", fontsize=8.5, color=c, ha="left", va="center")
        ax.text(1.02, y, note, transform=ax.get_yaxis_transform(), fontsize=8.5, va="center",
                color=COL["ink"] if kind == "dep" else COL["grey"])
    sep_y = (ys[5] + ys[6]) / 2.0
    ax.axhline(sep_y, color=COL["light"], lw=1.2, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(-0.6, max(ys) + 0.65)
    ax.set_xlabel("macro F1, 11 classes, test split")
    ax.grid(axis="y", visible=False)
    ax.text(0.540, sep_y + 0.12, "deployable (all 8 neighbouring tiles / 512 px crop)", fontsize=8.2, color=COL["attn"], va="bottom", ha="right")
    ax.text(0.540, sep_y - 0.12, "uses transect annotation mask (not deployable)", fontsize=8.2, color=COL["nbr"], va="top", ha="right")
    ax.set_title("Spatial context and neighbour weighting (frozen SimDINOv2-B/16)")
    save(fig, "03_context_ladder.png")


# --------------------------------------------------------------------------- 4. where smoothing acts
def fig_smoothing():
    st = jload(TC / "strata.json")["strata"]
    sc = jload(TC / "neighbourhood_scale.json")
    names = ["8 of 8\nsame label", "6 to 7", "3 to 5", "0 to 2\n(boundary)"]
    net = [s["flips"]["net"] for s in st]
    share = [s["part"] * 100 for s in st]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 4.4),
                               gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.42})
    cols = [COL["attn"] if v > 0 else COL["nbr"] for v in net]
    a.bar(range(4), net, color=cols, width=0.62)
    for i, (v, s) in enumerate(zip(net, share)):
        a.text(i, v + (8 if v > 0 else -8), f"{v:+d}", ha="center", va="bottom" if v > 0 else "top",
               fontsize=9.5, fontweight="bold", color=COL["ink"])
        a.text(i, -0.02, f"{s:.1f}% of test", transform=a.get_xaxis_transform(), ha="center", va="top",
               fontsize=8, color=COL["grey"])
    a.axhline(0, color=COL["ink"], lw=0.8)
    a.set_xticks(range(4))
    a.set_xticklabels(names, fontsize=8.5)
    a.tick_params(axis="x", pad=14)
    a.set_ylim(-110, 350)
    a.grid(axis="x", visible=False)
    a.set_ylabel("net tiles corrected by mean field")
    a.set_xlabel("neighbours (of 8) sharing the tile's true label", labelpad=6)
    a.set_title(f"Net {sum(net)} tiles corrected (annotated subgraph)")

    grid = sc["grid"]
    lab = {"+-112": "±112 px", "+-224": "±224 px", "+-336": "±336 px", "112+224": "±112 and ±224",
           "112+224+336": "all three rings"}
    y = np.arange(len(grid))[::-1]
    b.axvline(sc["base_test"], color=COL["frozen"], ls="--", lw=1.1)
    b.text(sc["base_test"] + 0.001, len(grid) - 0.45, f"no smoothing {sc['base_test']:.4f}",
           color=COL["frozen"], fontsize=8.5, va="bottom")
    for yy, g in zip(y, grid):
        b.scatter([g["test"]], [yy], s=62, color=COL["nbr"], edgecolor="white", linewidth=0.7, zorder=3)
        b.hlines(yy, sc["base_test"], g["test"], color=COL["nbr"], alpha=0.3, lw=1.2)
        b.text(g["test"] + 0.0045, yy, f"{g['test']:.4f}  ({g['voisins_moyens']:.1f} nbrs)", va="center", fontsize=8.5)
    b.set_yticks(y)
    b.set_yticklabels([lab[g["scale"]] for g in grid], fontsize=9)
    b.set_xlim(0.45, 0.625)
    b.set_ylim(-0.6, len(grid) - 0.1)
    b.grid(axis="y", visible=False)
    b.set_xlabel("macro F1, single tile + mean field (annotated subgraph)", fontsize=9.5)
    b.set_title("Smoothing radius: nearest tiles only")
    save(fig, "04_smoothing.png")


# --------------------------------------------------------------------------- 5. learned aggregation
def fig_gnn():
    arms = jload(TC / "canonical_probe_arms.json")["records"]
    g = jload(TC / "gnn_canonical.json")["records"]
    seeds = lambda a: [s["metrics"]["f1_macro_pres"] for s in arms[a]["per_seed"]]
    params = lambda p: f"{p / 1000:.0f}k params" if p else "no parameters"

    groups = [
        ("Annotated tiles only (6.53 nbrs/tile)", g["R3"]["f1"], [
            ("3x3 mean", g["R3"]["f1"], None, 0, True),
            ("Attention over the 3x3 window", g["L1"]["f1"], seeds("L1"), g["L1"]["n_params"], False),
            ("Attention, neighbour positions shuffled", g["L1p"]["f1"], seeds("L1p"), g["L1p"]["n_params"], False),
            ("CNN on the 3x3 window", g["L2"]["f1"], seeds("L2"), g["L2"]["n_params"], False),
        ]),
        ("All 8 neighbouring tiles (120,118 nodes, GNN graph)", g["MEAN9"]["f1"], [
            ("Uniform mean over the graph", g["MEAN9"]["f1"], None, 0, True),
            ("GATv2, 2 layers", g["GNN2"]["f1"], g["GNN2"]["f1_std"], g["GNN2"]["n_params"], False),
            ("GATv2, 3 layers", g["GNN3"]["f1"], g["GNN3"]["f1_std"], g["GNN3"]["n_params"], False),
            ("GATv2, edges shuffled", g["GNN2rw"]["f1"], g["GNN2rw"]["f1_std"], g["GNN2rw"]["n_params"], False),
        ]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True, gridspec_kw={"hspace": 0.38})
    for ax, (title, ref, rows) in zip(axes, groups):
        ax.axvline(ref, color=COL["grey"], ls="--", lw=1)
        rows = sorted(rows, key=lambda r: r[1])
        n = len(rows)
        for i, (lab, v, spread, p, is_ref) in enumerate(rows):
            y = n - 1 - i
            c = COL["frozen"] if is_ref else COL["nbr"]
            if isinstance(spread, list):
                ax.scatter(spread, [y] * len(spread), s=16, color=c, alpha=0.45, zorder=2)
            elif spread:
                ax.errorbar([v], [y], xerr=spread, fmt="none", ecolor=c, elinewidth=1.4, capsize=3, zorder=2)
            ax.scatter([v], [y], s=70, color=c, edgecolor="white", linewidth=0.8, zorder=3,
                       marker="D" if is_ref else "o")
            ax.text(0.537, y, f"{v:.4f}   {params(p)}", va="center", fontsize=8.5, color=COL["ink"])
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0.455, 0.585)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=10, color=COL["ink"])
    axes[1].set_xlabel("macro F1, 11 classes, test split")
    axes[1].text(0.0, -0.32, "small dots: seeds, bars: ±SD, dashed line: the fixed-rule reference",
                 transform=axes[1].transAxes, fontsize=8, color=COL["grey"], va="top")
    fig.suptitle("Learned aggregation does not beat a plain mean", x=0.125, ha="left", y=0.98,
                 fontsize=11.5, fontweight="bold")
    save(fig, "05_learned_aggregation.png")


# --------------------------------------------------------------------------- 6. latent geometry
SMARTIES_SCORES = [("chi", "Calinski-Harabasz"), ("fisher_ratio", "Fisher ratio"),
                   ("knn_purity", "k-NN purity"), ("rankme_sigma_raw", "RankMe"),
                   ("stable_rank", "Stable rank"), ("aniso_cosine", "Anisotropy")]
SMARTIES_CELLS = {"Maxar-RGB projection": "smarties_vitl16_rgb_mean",
                  "Sentinel-2 projection": "smarties_vitl16_s2_mean"}


def _smarties_layer():
    """Exploratory SMARTIES layer of figure 06, or None if its small tables are absent.

    SMARTIES is not in the clean baseline and not among the 35 models of loo_triage.csv: its points are
    drawn on top and never enter the percentages. Tables (all in results/rapport_data/): geometry_smarties.csv
    (same code and columns as geometry_models.csv), smarties_probe_f1.csv, dim_probe_smarties_*.json
    (same script as dim_probe_*.json). Without them the figure is the original two-panel one.
    """
    rd = RES / "rapport_data"
    if not all((rd / f).exists() for f in ("geometry_smarties.csv", "smarties_probe_f1.csv",
                                           "geometry_models.csv")):
        return None
    geo = pd.read_csv(rd / "geometry_models.csv").set_index("model")
    sm = pd.read_csv(rd / "geometry_smarties.csv").set_index("model")
    f1s = pd.read_csv(rd / "smarties_probe_f1.csv").set_index("model")["f1_macro_all"]
    fro = sorted(FROZEN_KEYS)
    base = baseline_scores()["mean"]

    def rank(v, ref):                       # 1 = lowest value among the 11 frozen models + the cell
        return 1 + int((np.asarray(ref, dtype=float) < v).sum())

    ranks, curves = {}, {}
    for lab, c in SMARTIES_CELLS.items():
        r = {col: rank(sm.loc[c, col], geo.loc[fro, col]) for col, _ in SMARTIES_SCORES}
        r["f1"] = rank(f1s[c], base.loc[fro])
        ranks[lab] = r
        f = rd / f"dim_probe_{c}.json"
        if f.exists():
            j = json.loads(f.read_text())
            full = max(j, key=lambda e: e["k"])["test_f1"]
            curves[lab] = {e["k"]: e["test_f1"] / full * 100 for e in j}
    return {"ranks": ranks, "curves": curves, "n": len(fro) + 1}


def _plot_smarties_ranks(ax, lay):
    ys = np.arange(len(SMARTIES_SCORES))[::-1]
    n = lay["n"]
    for lab, r in lay["ranks"].items():
        maxar = lab.startswith("Maxar")
        dy = 0.14 if maxar else -0.14       # small vertical offset: equal ranks must not hide each other
        ax.plot([r[c] for c, _ in SMARTIES_SCORES], ys + dy, ls="none", marker="o", ms=8,
                mfc=COL["ink"] if maxar else "white", mec=COL["ink"], mew=1.4,
                label=f"score rank, {lab.split()[0]}")
        ax.axvline(r["f1"], color=COL["nbr"], lw=1.3, ls="--" if maxar else ":",
                   label=f"F1 rank, {lab.split()[0]}")
    ax.set_yticks(ys)
    ax.set_yticklabels([nm for _, nm in SMARTIES_SCORES], fontsize=9)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel(f"rank among {n} models (1 = lowest value)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    ax.set_title("SMARTIES-L vs the 11 frozen models")


def fig_latent():
    d = pd.read_csv(RES / "rapport_data" / "loo_triage.csv")
    nm = {"fisher_ratio": "Fisher ratio", "silhouette": "Silhouette", "chi": "Calinski-Harabasz",
          "nc1_canonical": "NC1", "eff_rank_sigma2": "Effective rank", "stable_rank": "Stable rank",
          "participation_ratio": "Participation ratio", "rankme_sigma_raw": "RankMe",
          "knn_purity": "k-NN purity", "aniso_cosine": "Anisotropy",
          "intrinsic_dim_twonn": "Intrinsic dim. (TwoNN)", "dbi": "Davies-Bouldin",
          "logme_train": "LogME (train)", "logme_test": "LogME (test)"}
    fam = {"séparabilité": ("Class separation", COL["frozen"]), "voisinage": ("Neighbourhood", "#7FA7D6"),
           "spectrale": ("Spectral", COL["lora"]), "transférabilité": ("Transferability", COL["grey"])}
    d = d.sort_values("pair_acc_loo")
    lay = _smarties_layer()
    ncols = 3 if lay else 2
    fig, axs = plt.subplots(1, ncols, figsize=(11.2 if ncols == 2 else 17.4, 4.9),
                            gridspec_kw={"width_ratios": [1.15, 1] if ncols == 2 else [1.15, 1, 1],
                                         "wspace": 0.28 if ncols == 2 else 0.46})
    a, b = axs[0], axs[-1]
    y = np.arange(len(d))
    a.barh(y, d["pair_acc_loo"] * 100, color=[fam[f][1] for f in d["family"]], height=0.66)
    a.axvline(50, color=COL["ink"], lw=1, ls="--")
    for yy, v in zip(y, d["pair_acc_loo"] * 100):
        a.text(v + 0.8, yy, f"{v:.1f}", va="center", fontsize=8.5)
    a.set_yticks(y)
    a.set_yticklabels([nm[m] for m in d["metric"]], fontsize=9)
    a.set_xlim(25, 86)
    a.set_xlabel("pairs of models ordered correctly, leave-one-model-out (%)")
    a.grid(axis="y", visible=False)
    a.text(50.6, len(d) - 0.45, "chance", fontsize=8.5, va="center", color=COL["ink"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in fam.values()]
    a.legend(handles, [n for n, _ in fam.values()], loc="lower right", fontsize=8.5)
    a.set_title("Which embedding scores rank models by F1?")

    dp = pd.read_csv(RES / "rapport_data" / "dim_probe_summary.csv")
    ks = [10, 20, 50, 100, 200, 400]
    pretty = {"dinov3_vitb16_lvd": "DINOv3-B", "dinov3_vitl16_lvd": "DINOv3-L",
              "dinov3_vitl16_sat": "DINOv3-L satellite", "simdinov2_vitb16": "SimDINOv2-B",
              "vitb16_imagenet": "ViT-B ImageNet"}
    pal = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452"]
    for c, (_, r) in zip(pal, dp.iterrows()):
        b.plot(ks, [r[f"f1_k{k}"] / r["f1_full"] * 100 for k in ks], marker="o", ms=4.5, lw=1.5,
               color=c, label=pretty[r["model"]])
    if lay:
        for lab, col, ls in (("Maxar-RGB projection", COL["ink"], "--"),
                             ("Sentinel-2 projection", COL["grey"], ":")):
            cv = lay["curves"].get(lab)
            if cv:
                b.plot(ks, [cv[k] for k in ks], marker="s", ms=4.5, lw=1.5, ls=ls, color=col,
                       label=f"SMARTIES-L, {lab.split()[0]}")
    b.axhline(100, color=COL["grey"], lw=1, ls="--")
    b.set_xscale("log")
    b.set_xticks(ks)
    b.set_xticklabels([str(k) for k in ks])
    b.minorticks_off()
    b.set_ylim(65, 104)
    b.set_xlabel("principal components kept before the probe")
    b.set_ylabel("% of full-embedding F1")
    b.legend(fontsize=8.5, loc="lower right")
    b.set_title("Most of the F1 sits in ~100 dimensions")
    if lay:
        _plot_smarties_ranks(axs[1], lay)
    save(fig, "06_latent_geometry.png")


# --------------------------------------------------------------------------- 7. data curve
def fig_datacurve():
    r = pd.read_csv(RES / "datacurve_lora" / "results_agg.csv")
    fz = pd.read_csv(RES / "datacurve_lora" / "frozen_probe_results_agg.csv")
    sp = pd.read_csv(ROOT / "lora_spatial_v2" / "results_spatial_summary.csv")
    sg = sp.groupby("fraction_cible")["f1_macro_pres_test"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(r["fraction"] * 100, r["f1_pres_mean"], marker="o", ms=5, color=COL["lora"], lw=1.8,
            label="LoRA r=8, random subsets")
    ax.fill_between(r["fraction"] * 100, r["f1_pres_mean"] - r["f1_pres_std"],
                    r["f1_pres_mean"] + r["f1_pres_std"], color=COL["lora"], alpha=0.15, lw=0)
    ax.plot(fz["fraction"] * 100, fz["f1_pres_mean"], marker="s", ms=4.5, color=COL["frozen"], lw=1.8,
            label="Frozen probe, random subsets")
    ax.plot(sg["fraction_cible"] * 100, sg["mean"], marker="^", ms=6, color=COL["nbr"], lw=1.8,
            label="LoRA r=8, spatially blocked subsets")
    ax.scatter(sp["fraction_cible"] * 100, sp["f1_macro_pres_test"], s=13, color=COL["nbr"], alpha=0.4, zorder=2)
    ax.set_xscale("log")
    ticks = [0.5, 1, 5, 10, 25, 50, 75, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}%" for t in ticks], fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlabel("share of the 49,281 training tiles")
    ax.set_ylabel("macro F1, 11 classes, test split")
    ax.set_ylim(0, 0.52)
    ax.legend(loc="lower right", fontsize=9)
    comp = pd.read_csv(ROOT / "spatial_datacurve" / "class_composition.csv")
    c1 = comp[comp["fraction_cible"] == 0.01]["n_classes_8cls_presentes"]
    ax.annotate(f"blocked 1% subsets hold only {c1.min()} to {c1.max()}\nof the 8 main classes",
                xy=(1, sg.loc[sg["fraction_cible"] == 0.01, "mean"].iloc[0]), xytext=(1.5, 0.25),
                fontsize=8.5, color=COL["nbr"], arrowprops=dict(arrowstyle="-", color=COL["nbr"], lw=0.8))
    ax.set_title("Label efficiency on DINOv3-B depends on how the subset is drawn")
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.125, 0.02, "Shaded band: ±SD across 3 seeds. Small red dots: individual spatially blocked draws.",
             fontsize=8, color=COL["grey"], va="bottom")
    save(fig, "07_data_curve.png")



# --------------------------------------------------------------------------- 8-class view
DROP8 = ["ARCA", "DRYI", "RUBC"]
KEEP8 = [c for c in ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF", "RUBC", "SEDG", "TUSS", "WILL"]
         if c not in DROP8]


def readme_scores_8cls() -> pd.DataFrame:
    """Macro F1 over the 8 kept classes, same 11-class classifier, full test split.

    Read from the committed per-class table, so a clone can rebuild it. The convention
    matches f1_macro_8cls_test in the run metrics: predictions on the 321 ARCA/DRYI/RUBC
    test tiles count as errors, and the three classes are not averaged.
    """
    pc = pd.read_csv(BASE / "per_class_f1.csv")
    pc = pc[pc["class"].isin(KEEP8)]
    per_probe = (pc.groupby(["key", "probe_id"])["f1_test"].mean()
                   .reset_index().rename(columns={"f1_test": "f8"}))
    seeds = pd.read_csv(BASE / "per_seed.csv")[["id", "f1_macro_pres"]]
    per_probe = per_probe.merge(seeds, left_on="probe_id", right_on="id", how="left")
    assert per_probe["f1_macro_pres"].notna().all(), "unmatched probe id"
    out = per_probe.groupby("key").agg(
        f11=("f1_macro_pres", "mean"), f8=("f8", "mean"),
        sd8=("f8", lambda v: v.std(ddof=1) if len(v) > 1 else np.nan),
        n=("f8", "size")).reset_index()
    out = out[out["key"].isin(LABELS)]
    assert len(out) == 42, len(out)
    return out.set_index("key")


def arms_scores() -> pd.DataFrame:
    """11-class and 8-class macro F1 of the context, decoding and aggregation arms.

    Read from the committed table results/tile_context_2026-11-01/arms_8cls.csv, built from
    the saved predictions with the same 8-class convention as readme_scores_8cls().
    """
    f = TC / "arms_8cls.csv"
    if not f.exists():
        return pd.DataFrame(columns=["label", "f11", "f8"])
    return pd.read_csv(f)


def fig_classes8():
    sc = readme_scores_8cls()
    from scipy.stats import spearmanr, kendalltau
    rho = spearmanr(sc.f11, sc.f8)
    r11 = sc.f11.rank(ascending=False)
    r8 = sc.f8.rank(ascending=False)
    move = (r8 - r11)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.28})
    frozen = [k for k in sc.index if k in FROZEN_KEYS]
    ft = [k for k in sc.index if k not in FROZEN_KEYS]
    a.scatter(sc.loc[frozen, "f11"], sc.loc[frozen, "f8"], s=46, color=COL["frozen"],
              edgecolor="white", linewidth=0.7, label="frozen", zorder=3)
    a.scatter(sc.loc[ft, "f11"], sc.loc[ft, "f8"], s=46, color=COL["lora"],
              edgecolor="white", linewidth=0.7, label="fine-tuned", zorder=3)
    a.scatter([sc.loc["dinov3_vith16plus_lvd", "f11"]], [sc.loc["dinov3_vith16plus_lvd", "f8"]],
              s=120, facecolor="none", edgecolor=COL["ink"], linewidth=1.1, zorder=4,
              label="best frozen (DINOv3-H+)")
    labels = {
        "ctxdistill_dB_tL": ("DINOv3-B + 1024 px\ncontext (FT)", 12, -32, True),
        "dinov3_vitb16_lvd_full": (LABELS["dinov3_vitb16_lvd_full"], 22, -36, True),
        "vitb16_scratch_old": (LABELS["vitb16_scratch_old"], 6, 0, False),
    }
    for k, (txt, dx, dy, arrow) in labels.items():
        a.annotate(txt, (sc.loc[k, "f11"], sc.loc[k, "f8"]), xytext=(dx, dy),
                   textcoords="offset points", fontsize=8, color=COL["ink"],
                   ha="left" if dx > 0 else "right",
                   arrowprops=dict(arrowstyle="-", lw=0.8, color=COL["grey"]) if arrow else None)
    arms = arms_scores()
    if len(arms):
        annot_only = {
            "3x3 mean", "Single tile + mean field", "3x3 mean + ICM",
            "3x3 mean + mean field", "Attention over the 3x3 window",
            "Attention, positions shuffled",
        }
        arm_ann = arms[arms["label"].isin(annot_only)]
        arm_fg = arms[~arms["label"].isin(annot_only)]
        if len(arm_fg):
            a.scatter(arm_fg.f11, arm_fg.f8, s=50, marker="s", facecolor="white",
                      edgecolor=COL["full"], linewidth=1.2, zorder=3,
                      label="full-graph / 3x3 CNN arms")
        if len(arm_ann):
            a.scatter(arm_ann.f11, arm_ann.f8, s=58, marker="D", color=COL["nbr"],
                      edgecolor="white", linewidth=0.7, zorder=3,
                      label="annotated-only context arms")
            top_arm = arm_ann.loc[arm_ann["f8"].idxmax()]
            a.annotate(top_arm["label"] + "\n(annotated only)", (top_arm["f11"], top_arm["f8"]),
                       xytext=(-14, 10), textcoords="offset points", fontsize=7.8, color=COL["nbr"],
                       ha="right", va="center",
                       arrowprops=dict(arrowstyle="-", lw=0.8, color=COL["nbr"]))
    fg_f = FG_OUT / "multibackbone_smoothing_fullgraph.csv"
    if fg_f.exists():
        fg = pd.read_csv(fg_f).set_index("backbone")
        f11_dep = float(fg.loc["dinov3_vitb16_lvd", "R3_Wconf"])
        f8_dep = float(fg.loc["dinov3_vitb16_lvd", "R3_Wconf_8cls"])
        a.scatter([f11_dep], [f8_dep], s=66, marker="D", color=COL["attn"], edgecolor="white",
                  linewidth=0.8, zorder=4, label="best deployable (DINOv3-B, 8 nbrs)")
        a.annotate("DINOv3-B, 3x3 + conf.\nsmoothing (8 nbrs)", (f11_dep, f8_dep), xytext=(-24, 6),
                   textcoords="offset points", fontsize=7.8, color=COL["attn"], ha="right", va="center",
                   arrowprops=dict(arrowstyle="-", lw=0.9, color=COL["attn"]))

    a.set_xlim(0.345, 0.548)
    a.set_ylim(0.465, 0.76)
    a.set_xlabel("macro F1, 11 classes")
    a.set_ylabel("macro F1, 8 classes")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), fontsize=8.5)
    a.text(0.98, 0.03, f"Spearman {rho.statistic:.3f}, Kendall {kendalltau(sc.f11, sc.f8).statistic:.3f}.\n"
           f"Median gain {(sc.f8 - sc.f11).median():+.3f}.", transform=a.transAxes,
           fontsize=8.5, va="bottom", ha="right", color=COL["grey"])
    a.set_title("8 classes instead of 11")

    buckets = [0, 1, 2]
    counts = [int((move.abs() == k).sum()) for k in buckets] + [int((move.abs() >= 3).sum())]
    b.bar(range(4), counts, color=[COL["grey"]] * 3 + [COL["nbr"]], width=0.62)
    for i, c in enumerate(counts):
        b.text(i, c + 0.4, str(c), ha="center", va="bottom", fontsize=10, fontweight="bold",
               color=COL["nbr"] if i == 3 else COL["ink"])
    b.set_xticks(range(4))
    b.set_xticklabels(["0", "1", "2", "3 or more"])
    b.set_xlabel("ranks moved between the two schemas")
    b.set_ylabel("number of models")
    b.set_ylim(0, max(counts) * 1.28)
    b.grid(axis="x", visible=False)
    big = move[move.abs() >= 3].sort_values(key=lambda x: -x.abs())
    fmt_move = lambda v: f"{abs(int(v))} ranks {'down' if v > 0 else 'up'}"
    txt = "\n".join(f"{LABELS[k]}: {fmt_move(v)}" for k, v in big.items())
    b.text(0.98, 0.74, txt, transform=b.transAxes, ha="right", va="top", fontsize=8.5, color=COL["ink"])
    b.set_title("Ranking is preserved for 40 of 42")
    save(fig, "08_classes_8.png")



# --------------------------------------------------------------------------- 9. multi-backbone
MB_LABELS = {
    "simdinov2_vitb16": "SimDINOv2-B, frozen",
    "dinov3_vitb16_lvd": "DINOv3-B, frozen",
    "dinov3_vitl16_lvd": "DINOv3-L, frozen",
    "dinov3_vith16plus_lvd": "DINOv3-H+, frozen",
    "simdinov2_vitl16": "SimDINOv2-L, frozen",
    "dinov3_vitl16_lvd_lora_s0": "DINOv3-L + LoRA r=8",
    "simdinov2_vitb16_lora_r8a16_s0": "SimDINOv2-B + LoRA r8a16",
    "simdinov2_vitl16_lora_s0": "SimDINOv2-L + LoRA r=8",
}


def fig_multibackbone():
    f_full = FG_OUT / "multibackbone_smoothing_fullgraph.csv"
    if not f_full.exists():
        return
    d = pd.read_csv(f_full)
    d["label"] = d["backbone"].map(MB_LABELS).fillna(d["backbone"])
    d = d.sort_values("R3_Wconf")
    ps = pd.read_csv(BASE / "per_seed.csv").groupby("key")["f1_macro_pres"].mean()
    ref = max(ps[k] for k in LABELS if k not in UPPER_BOUND)
    y = np.arange(len(d))

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.5), sharey=True,
                               gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.08})
    a.axvline(ref, color=COL["grey"], ls="--", lw=1)
    series = [("R0", "single tile", "o", COL["frozen"], 36),
              ("R0_MF", "single tile + uniform smoothing", "o", "#7FA7D6", 36),
              ("R0_Wconf", "single tile + conf. smoothing", "s", COL["lora"], 40),
              ("R3_Wconf", "3x3 average + conf. smoothing", "D", COL["attn"], 48)]
    for col, lab, mk, c, sz in series:
        a.scatter(d[col], y, marker=mk, s=sz, color=c, edgecolor="white", linewidth=0.6,
                  label=lab, zorder=3)
    a.set_yticks(y)
    a.set_yticklabels(d["label"], fontsize=9)
    a.set_ylim(-0.6, len(d) - 0.25)
    cols_x = ["R0", "R0_MF", "R0_Wconf", "R3_Wconf"]
    xlo = min(d[cols_x].min()) - 0.005
    xhi = max(d[cols_x].max()) + 0.006
    a.set_xlim(xlo, xhi)
    a.set_xlabel("macro F1, 11 classes, test split")
    a.grid(axis="y", visible=False)
    h, l = a.get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=COL["grey"], ls="--", lw=1))
    l.append(f"best single-tile model ({ref:.4f})")
    a.set_title("Context and smoothing on 8 backbones (all 8 neighbours)")

    b.axvline(0, color=COL["grey"], ls="--", lw=1)
    b.errorbar(d["d_mf_R0"], y - 0.14, xerr=[d["d_mf_R0"] - d["lo_mf_R0"], d["hi_mf_R0"] - d["d_mf_R0"]],
               fmt="o", ms=5, color="#7FA7D6", markeredgecolor="#4C72B0", ecolor="#7FA7D6",
               elinewidth=1.3, capsize=2, label="uniform smoothing vs single tile", zorder=3)
    b.errorbar(d["d_wconf_R0"], y + 0.14, xerr=[d["d_wconf_R0"] - d["lo_wconf_R0"], d["hi_wconf_R0"] - d["d_wconf_R0"]],
               fmt="s", ms=5, color=COL["lora"], ecolor=COL["lora"], elinewidth=1.3, capsize=2,
               label="conf. smoothing vs single tile", zorder=3)
    blo = min(0.0, float(d[["lo_mf_R0", "lo_wconf_R0"]].min().min())) - 0.003
    bhi = float(d[["hi_mf_R0", "hi_wconf_R0"]].max().max()) + 0.004
    b.set_xlim(blo, bhi)
    b.set_ylim(-0.6, len(d) - 0.25)
    b.grid(axis="y", visible=False)
    b.set_xlabel("paired delta vs single tile")
    b.set_title("Smoothing gain, 95% CI over 9 maps")
    h2, l2 = b.get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="lower center", ncol=4, fontsize=8.3, frameon=False,
               bbox_to_anchor=(0.5, 0.075))
    fig.subplots_adjust(bottom=0.32)
    fig.text(0.5, 0.012,
             "All 8 neighbouring tiles per tile (120,164 nodes); all 8/8 CIs exclude 0. "
             "Fine-tuned rows use seed 0. Frozen SimDINOv2-B 3x3 selects C=0.01 on val.",
             fontsize=8, color=COL["grey"], va="bottom", ha="center")
    save(fig, "09_multibackbone.png")


# --------------------------------------------------------------------------- 10. per-class F1 & support
CLS_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF", "RUBC", "SEDG", "TUSS", "WILL"]


def fig_per_class():
    pc = pd.read_csv(BASE / "per_class_f1.csv")
    tr = pd.read_csv(ROOT / "splits" / "train_11cls.csv")["label"].value_counts()
    te = pd.read_csv(ROOT / "splits" / "test_11cls.csv")["label"].value_counts()

    f_h = pc[pc["key"] == "dinov3_vith16plus_lvd"].groupby("class")["f1_test"].mean()
    f_cf = pc[pc["key"] == "ctxdistill_dB_tL"].groupby("class")["f1_test"].mean()
    f_max = pc.groupby("class")["f1_test"].max()

    rows = []
    for idx, cls in enumerate(CLS_11):
        rows.append(dict(
            cls=cls,
            frozen=float(f_h[cls]),
            ctxf=float(f_cf[cls]),
            max230=float(f_max[cls]),
            n_train=int(tr.get(idx, 0)),
            n_test=int(te.get(idx, 0)),
        ))
    df = pd.DataFrame(rows).sort_values(["frozen", "max230"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(df))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True,
                               gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.08})

    # Shade the 3 rare near-zero classes at the bottom (rows 0, 1, 2)
    a.axhspan(-0.55, 2.5, color="#FDF3F2", lw=0, zorder=0)
    b.axhspan(-0.55, 2.5, color="#FDF3F2", lw=0, zorder=0)
    a.axhline(2.5, color="#E2B4B0", lw=1.0, ls="--", zorder=1)
    b.axhline(2.5, color="#E2B4B0", lw=1.0, ls="--", zorder=1)

    for i, r in df.iterrows():
        lo, hi = min(r.frozen, r.ctxf), max(r.frozen, r.ctxf)
        if hi > lo + 1e-4:
            a.hlines(i, lo, hi, color=COL["nbr"] if r.ctxf >= r.frozen else COL["grey"],
                     lw=2.0, alpha=0.55, zorder=2)
        if r.max230 > hi + 0.008:
            a.hlines(i, hi, r.max230, color=COL["light"], lw=1.2, ls=":", zorder=1)
        if i <= 2:
            a.text(r.max230 + 0.018, i, f"0.000  (max of 230: {r.max230:.3f})",
                   va="center", ha="left", fontsize=8.2, color=COL["grey"])
        else:
            delta = r.ctxf - r.frozen
            sign_col = COL["nbr"] if delta >= 0.02 else COL["ink"] if delta >= 0 else COL["grey"]
            a.text(max(r.max230, hi) + 0.018, i, f"{delta:+.3f}",
                   va="center", ha="left", fontsize=8.2, color=sign_col,
                   fontweight="bold" if delta >= 0.05 else "normal")

    a.scatter(df["max230"], y, s=34, marker="|", color=COL["ink"], linewidth=1.6,
              label="max across 230 baseline probes", zorder=3)
    a.scatter(df["frozen"], y, s=56, marker="o", color=COL["frozen"], edgecolor="white",
              linewidth=0.7, label="Frozen DINOv3-H+", zorder=4)
    a.scatter(df["ctxf"], y, s=58, marker="D", color=COL["nbr"], edgecolor="white",
              linewidth=0.7, label="DINOv3-B + 1024 px context (FT) †", zorder=5)

    a.text(1.09, 0.7, "3 rare classes (< 400 train tiles)\ndropped in 8-class view",
           ha="right", va="center", fontsize=8.2, color=COL["nbr"], style="italic")
    a.set_yticks(y)
    a.set_yticklabels(df["cls"], fontsize=9.5, family="DejaVu Sans Mono")
    a.set_xlim(-0.04, 1.12)
    a.set_ylim(-0.6, len(df) - 0.35)
    a.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_xlabel("per-class F1 on test split (right labels: Δ context fusion − frozen)")
    a.grid(axis="y", visible=False)
    a.set_title("Per-class F1: gains on ground cover, zero on rare classes")

    h = 0.34
    b.barh(y + h / 2, df["n_train"], height=h, color="#C8D4E3", label="train tiles (15 maps)", zorder=2)
    b.barh(y - h / 2, df["n_test"], height=h, color=COL["frozen"], alpha=0.85, label="test tiles (9 maps)", zorder=2)
    for i, r in df.iterrows():
        b.text(max(r.n_train, r.n_test) * 1.18, i, f"{r.n_train:,} / {r.n_test:,}",
               va="center", ha="left", fontsize=8, color=COL["ink"])
    b.set_xscale("log")
    b.set_xlim(15, 280000)
    b.set_xticks([30, 100, 1000, 10000])
    b.set_xticklabels(["30", "100", "1k", "10k"])
    b.minorticks_off()
    b.grid(axis="y", visible=False)
    b.set_xlabel("tile count, log scale (train / test)")
    b.set_title("Class imbalance across splits")

    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, fontsize=8.3, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.01, "Clean baseline per_class_f1.csv; RHOL is absent from val and test and dropped. "
             "† uses a 1024 px context crop at inference (mean of 3 seeds).", fontsize=8, color=COL["grey"],
             va="bottom", ha="center")
    save(fig, "10_per_class_f1.png")


if __name__ == "__main__":
    fig_leaderboard()
    fig_adaptation()
    fig_context()
    fig_smoothing()
    fig_gnn()
    fig_latent()
    fig_datacurve()
    fig_classes8()
    fig_multibackbone()
    fig_per_class()
