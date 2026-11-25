# Arctic-TVC: foundation models for tundra vegetation mapping

[![CI](https://github.com/Lmague/benchmark-memoire/actions/workflows/ci.yml/badge.svg)](https://github.com/Lmague/benchmark-memoire/actions/workflows/ci.yml)
[![Dataset DOI](https://img.shields.io/badge/Arctic--TVC-10.20383%2F103.01418-teal.svg)](https://doi.org/10.20383/103.01418)

Code and results for my M.Sc. thesis in computer science (AI) at UQAM, supervised by Mohamed Bouguessa (ML) and Étienne Laliberté (ecology).

The goal is to classify 224 px tiles cropped from Arctic tundra drone orthomosaics (about 2.2 mm/px) into 11 vegetation classes. I compare frozen foundation models, parameter-efficient fine-tuning (PEFT), full fine-tuning, and a from-scratch baseline under one shared linear probe and one orthomosaic-level test split. The main question is simple: on this kind of fine-scale vegetation map, what actually helps the most — a bigger/better backbone, fine-tuning, or using the neighbouring tiles around each tile?

Main takeaways:

- Across the DINOv3 and SimDINOv2 models, changing model size or fine-tuning moves macro F1 by at most 0.015. With 9 test orthomosaics, none of these models is significantly different from the best frozen backbone. Only the weakest baselines stand out: SatMAE, frozen ResNet-50 ImageNet, and ViT-B/16 trained from scratch.
- Using neighbouring tiles and spatial smoothing increases macro F1 by roughly +0.04 to +0.06. The smoothing step requires no training; only the linear probe is refit on the averaged features.
- Learned neighbour aggregation (self-attention, CNN, GATv2) does not beat a simple uniform average.
- Embedding geometry metrics can rank models slightly better than chance, but they do not predict F1 accurately.

My highest score overall is **0.5394 macro F1**, obtained with a frozen DINOv3-L backbone, a 3x3 average of tile embeddings, and mean-field smoothing on the predicted map (in the 8-backbone experiment with extended smoothing grids). For the SimDINOv2-B pipeline analyzed in detail below, the same recipe reaches 0.5359 in its dedicated run (0.5362 in the 8-backbone run), compared to 0.4720 on a single tile in that run.

Note that these multi-tile recipes use neighbouring tiles at inference time, so they are evaluated separately from the 42 single-input configurations. Among deployable single-tile models, the best one is context distillation at 0.4872. The three context-fusion rows (0.5052 to 0.5079) require both the tile and a 512 px context patch at inference time, so I treat them as non-deployable upper bounds (same for the frozen tile + 512 px recipe in the figure).

Contents: [Ranking](#ranking-of-42-configurations) · [Adaptation](#adaptation-methods) · [Context and smoothing](#neighbouring-tiles-and-smoothing) · [Other backbones](#does-it-hold-on-other-backbones) · [Learned aggregation](#learned-aggregation) · [Embedding geometry](#embedding-geometry) · [Data size](#data-size) · [Per class](#per-class) · [Protocol](#protocol) · [Ongoing](#ongoing-work) · [Limits](#limits) · [Reproduce](#reproduce)

## Ranking of 42 configurations

![Named ranking and significance](figures/readme/01_leaderboard.png)

This ranking covers 42 configurations: 11 frozen models and 31 fine-tuned models (3 seeds each). Out of the 42, 39 are deployable on a single tile; the top 3 rows (marked "upper bound" in the table) need both the tile and a wider context window at inference. This corresponds to the canonical tier of the clean baseline plus those 3 upper bounds; the other 112 runs registered in the baseline (supporting and exploratory experiments) are not included in this ranking. Two supporting runs actually score slightly higher than the best canonical single-tile model and are kept out of this table: a DINOv3-B rank ablation (0.4845) and a single-seed spatial LoRA run (0.4885). Every model is evaluated with the exact same linear probe. The figure groups the results into three input levels: single tile (top 18 models plus 8 reference models, including frozen DINOv3-B), neighbouring tiles as input, and map-level smoothing. The full 42-model table is folded below.

- Best frozen model: DINOv3 ViT-H+/16 at 0.4805. Best standard LoRA in this ranking: DINOv3-L at 0.4842. The difference is +0.0037 (CI [-0.034, +0.017], q = 0.76), which is not statistically significant.
- After Benjamini-Hochberg correction, only 3 of the 38 single-tile models are significantly different from the best frozen model: SatMAE (-0.071), frozen ResNet-50 ImageNet (-0.073), and ViT-B/16 from scratch (-0.126). All other pairs fail the correction (q > 0.10), even though three of them still have paired 95% CIs excluding zero (ViT-B/16 ImageNet, Scale-MAE, and fine-tuned ResNet-50).
- Satellite-pretrained DINOv3 ViT-L gets 0.4620 vs 0.4791 for the natural-image version of the same architecture (-0.017, CI [-0.058, +0.001], not significant). Scale-MAE (0.4478) and SatMAE (0.4097) are both ViT-L models and still score below frozen ViT-B/16 ImageNet (0.4497), though the architecture size differs so this alone does not isolate the effect of satellite pretraining.
- Adding fine-tuning on top of the 512 px context fusion does not seem to help: the two LoRA variants get 0.5056 and 0.5052 (clean baseline) compared to 0.5058 for the frozen fusion (tile-context run). Since those numbers come from separate runs, this is indicative rather than a paired test.
- The context-fusion and 3x3-mean setups (0.5052 to 0.5205) are above every single-tile model (three of them are non-deployable upper bounds). By contrast, learned aggregation does not reach that level: the 3x3 CNN gets 0.4824 and GATv2 with shuffled edges gets 0.4648, both below the best single-tile model. These setups are outside the Benjamini-Hochberg test family above.
- Non-linear classification heads (MLP, bilinear, FiLM) never beat the linear probe on single-tile features (0 out of 33 groups, mean delta -0.0103). They do slightly better on 2 out of 24 fused groups, both single-seed and without confidence intervals, so I do not count them as reliable gains.

<details>
<summary>All 42 configurations</summary>

| # | Model | Macro F1, 11 classes | Macro F1, 8 classes | Seed SD (11 / 8) | Runs | Deployable |
|---:|---|---:|---:|---:|---:|---|
| 1 | Context fusion, fine-tuned | 0.5079 | 0.6984 | 0.0015 / 0.0020 | 3 | upper bound |
| 2 | Context fusion, SimDINOv2-B LoRA r8a16 | 0.5056 | 0.6953 | 0.0071 / 0.0098 | 3 | upper bound |
| 3 | Context fusion, SimDINOv2-B LoRA r2a4 | 0.5052 | 0.6947 | 0.0030 / 0.0042 | 3 | upper bound |
| 4 | Context distillation, tile only at inference | 0.4872 | 0.6699 | 0.0018 / 0.0025 | 3 | yes |
| 5 | DINOv3-L, LoRA r=8 | 0.4842 | 0.6657 | 0.0012 / 0.0017 | 3 | yes |
| 6 | Context distillation, EMA teacher, tile only | 0.4835 | 0.6648 | 0.0017 / 0.0024 | 3 | yes |
| 7 | SimDINOv2-L, LoRA r=8 | 0.4835 | 0.6633 | 0.0068 / 0.0106 | 3 | yes |
| 8 | DINOv3-B, LoRA r=8 | 0.4830 | 0.6641 | 0.0037 / 0.0051 | 3 | yes |
| 9 | SimDINOv2-B, LoRA r=8, Q+K+V | 0.4812 | 0.6617 | 0.0052 / 0.0072 | 3 | yes |
| 10 | SimDINOv2-B, LoRA r=8, blocks 9-11 | 0.4805 | 0.6608 | 0.0001 / 0.0001 | 3 | yes |
| 11 | DINOv3 ViT-H+/16, frozen | 0.4805 | 0.6606 | n/a / n/a | 1 | yes |
| 12 | SimDINOv2-B, LoRA r=8, blocks 6-11 | 0.4803 | 0.6604 | 0.0028 / 0.0038 | 3 | yes |
| 13 | SimDINOv2-B, LoRA r=2 | 0.4793 | 0.6591 | 0.0024 / 0.0033 | 3 | yes |
| 14 | DINOv3 ViT-L/16, frozen | 0.4791 | 0.6588 | n/a / n/a | 1 | yes |
| 15 | DINOv3-B, attention only | 0.4790 | 0.6572 | 0.0025 / 0.0047 | 3 | yes |
| 16 | DINOv3-B, full fine-tuning | 0.4788 | 0.6535 | 0.0025 / 0.0032 | 3 | yes |
| 17 | SimDINOv2-B, LoRA r=4 | 0.4783 | 0.6577 | 0.0050 / 0.0069 | 3 | yes |
| 18 | SimDINOv2-B, norm layers + head only | 0.4783 | 0.6577 | 0.0002 / 0.0003 | 3 | yes |
| 19 | SimDINOv2-B, full fine-tuning | 0.4782 | 0.6575 | 0.0074 / 0.0102 | 3 | yes |
| 20 | SimDINOv2-B, LoRA r=8 | 0.4780 | 0.6572 | 0.0029 / 0.0039 | 3 | yes |
| 21 | DINOv3-S, LoRA r=8 | 0.4776 | 0.6567 | 0.0023 / 0.0031 | 3 | yes |
| 22 | SimDINOv2-B, LoRA r=8, blocks 0-5 | 0.4771 | 0.6560 | 0.0023 / 0.0032 | 3 | yes |
| 23 | SimDINOv2-B, LoRA r=16 | 0.4769 | 0.6557 | 0.0015 / 0.0021 | 3 | yes |
| 24 | SimDINOv2-B, LoRA r=8, scaling 2 | 0.4761 | 0.6547 | 0.0017 / 0.0023 | 3 | yes |
| 25 | SimDINOv2 ViT-L/16, frozen | 0.4756 | 0.6540 | n/a / n/a | 1 | yes |
| 26 | SimDINOv2-B, LoRA r=8, rsLoRA | 0.4751 | 0.6533 | 0.0020 / 0.0027 | 3 | yes |
| 27 | SimDINOv2-B, LoRA r=32 | 0.4751 | 0.6532 | 0.0018 / 0.0025 | 3 | yes |
| 28 | SimDINOv2-B, LoRA r=16, scaling 2 | 0.4750 | 0.6531 | 0.0026 / 0.0036 | 3 | yes |
| 29 | ViT-B/16 ImageNet, full fine-tuning | 0.4748 | 0.6513 | 0.0015 / 0.0016 | 3 | yes |
| 30 | SimDINOv2-B, LoRA r=16, rsLoRA | 0.4735 | 0.6497 | 0.0035 / 0.0034 | 3 | yes |
| 31 | SimDINOv2 ViT-B/16, frozen | 0.4719 | 0.6488 | n/a / n/a | 1 | yes |
| 32 | DINOv3 ViT-B/16, frozen | 0.4715 | 0.6483 | n/a / n/a | 1 | yes |
| 33 | SimDINOv2-B, attention only (MHSA) | 0.4692 | 0.6430 | 0.0022 / 0.0031 | 3 | yes |
| 34 | DINOv3 ViT-S/16, frozen | 0.4689 | 0.6448 | n/a / n/a | 1 | yes |
| 35 | ViT-B/16 ImageNet, attention only | 0.4686 | 0.6430 | 0.0048 / 0.0082 | 3 | yes |
| 36 | DINOv3 ViT-L/16 satellite, frozen | 0.4620 | 0.6352 | n/a / n/a | 1 | yes |
| 37 | ResNet-50 ImageNet, full fine-tuning | 0.4573 | 0.6287 | 0.0033 / 0.0046 | 3 | yes |
| 38 | ViT-B/16 ImageNet, frozen | 0.4497 | 0.6163 | n/a / n/a | 1 | yes |
| 39 | Scale-MAE ViT-L/16, frozen | 0.4478 | 0.6157 | n/a / n/a | 1 | yes |
| 40 | SatMAE ViT-L/16, frozen | 0.4097 | 0.5633 | n/a / n/a | 1 | yes |
| 41 | ResNet-50 ImageNet, frozen | 0.4077 | 0.5606 | n/a / n/a | 1 | yes |
| 42 | ViT-B/16 from scratch | 0.3547 | 0.4805 | 0.0115 / 0.0067 | 3 | yes |

</details>

## Adaptation methods

![Adaptation regimes](figures/readme/02_adaptation_regimes.png)

All 16 adaptation methods tested on SimDINOv2-B/16 end up between 0.4692 and 0.4812, compared to 0.4719 for the frozen backbone. I use SimDINOv2-B as the main testbed here because it is the only backbone where I ran the full 16-regime ablation in the clean baseline. Its frozen F1 is almost identical to DINOv3-B (0.4715, a 0.0004 gap), and the 8-backbone check below shows the same context and smoothing gains on DINOv3-B (+0.055 up to 0.5366, vs +0.057 up to 0.5362 for SimDINOv2-B). None of the 16 adapted versions is significantly better than the frozen model (the best one is LoRA r=8 on Q+K+V at +0.0094, CI [-0.006, +0.020], q = 0.24).

- Going from LoRA rank 2 to rank 32 drops the mean score from 0.4793 to 0.4751, but no pair is statistically different (rank 2 vs rank 32: +0.0042, CI [-0.003, +0.009], q = 0.31), so I cannot confirm a real downward trend.
- Updating only the normalization layers and the linear head gives 0.4783, practically indistinguishable from LoRA r=8 (CI [-0.009, +0.009]); compared to full fine-tuning, the CI is still wide enough to allow up to a +0.021 difference.
- Full fine-tuning is the most sensitive to the random seed: 0.4696 to 0.4827 across 3 seeds.
- Tuning only attention layers (MHSA) drops to 0.4692, which is 0.0027 below frozen, though again not statistically significant (CI [-0.020, +0.008], q = 0.78).

For perspective, the top part of the figure plots the neighbourhood and smoothing setups on the exact same frozen backbone. Compared to 0.4719 frozen and 0.4812 for the best adaptation, the 512 px context gets 0.5058, the 3x3 average reaches 0.5205, and adding mean-field smoothing reaches 0.5359. GATv2 on the full graph gets 0.4933, still below the simple 3x3 average. Overall, context and smoothing bring roughly seven times more gain than adaptation (+0.064 vs +0.009).

## Neighbouring tiles and smoothing

![Context ladder](figures/readme/03_context_ladder.png)

All rows here use the same frozen SimDINOv2-B/16 embeddings and the canonical linear probe. Paired differences are computed with a hierarchical bootstrap (resampling orthomosaics first, then tiles) over 10,000 replicates.

- Adding neighbouring tiles increases F1 from 0.4718 to 0.5205. Moving from tile + 512 px context to the 3x3 average is not statistically significant on its own: paired-bootstrap delta +0.0136 (point difference 0.0147), CI [-0.032, +0.036], p = 0.20.
- Mean-field smoothing gives the largest single gain: +0.0578 on top of a single tile, and +0.0152 on top of the 3x3 average. Both CIs exclude zero (note that I did not run a direct paired test comparing those two gains against each other).
- Once smoothing is applied, the difference between the 3x3 average and a single tile is no longer significant (+0.0062, CI [-0.010, +0.053], p = 0.28), although the CI still leaves room for a gain up to +0.05.
- Mean-field smoothing has no learned weights and requires no training. Its strength (beta) and number of iterations are picked on the validation maps. In these runs, both the iteration count and the ±112 px strength hit the upper bound of the search grid, so the reported gains might actually be slightly conservative. Neighbour embeddings are loaded from disk cache.
- Concatenating classical colour and texture descriptors to the tile embedding hurts F1 by 0.004 to 0.005 in both runs (no paired CI computed).

![Where smoothing acts](figures/readme/04_smoothing.png)

- Out of the 349 net tiles corrected by mean-field smoothing, 301 are in mixed neighbourhoods (where 3 to 5 out of 8 neighbours share the tile's true class). As expected, tiles sitting right on class boundaries lose accuracy (-55 net tiles).
- For mean-field smoothing, using only the closest ring of neighbours works best (tiles are 224 px with a 112 px stride, so ±112 px corresponds to 50% overlap): 0.5294, compared to 0.5102 at ±224 px and 0.4957 at ±336 px. Validation follows the same order (0.4891 vs 0.4737 at ±224 px). Combining multiple rings gives 0.5195 to 0.5228. I did not compute CIs for these radius comparisons, so treat them as point estimates.

## Does it hold on other backbones?

![Context and smoothing on eight backbones](figures/readme/09_multibackbone.png)

To make sure this was not specific to SimDINOv2-B, I ran the same split, probe, and tile graph on eight backbones: five frozen (SimDINOv2-B, SimDINOv2-L, DINOv3-B, DINOv3-L, DINOv3-H+) and three LoRA fine-tuned models (DINOv3-L r=8, SimDINOv2-B r8a16, SimDINOv2-L r=8, all seed 0). Here I used wider smoothing grids on validation (beta from 0.1 to 16, 1 to 12 iterations). The SimDINOv2-B row matches the dedicated decoding run to within 0.0001 / 0.0000 / 0.0003 (0.4720 / 0.5207 / 0.5362 vs 0.4719 / 0.5207 / 0.5359; the probe-arms run gave 0.4718 / 0.5205).

| Backbone | Single tile | + mean field | 3x3 mean | 3x3 + mean field | Mean field on 3x3, 95% CI |
|---|---:|---:|---:|---:|---|
| DINOv3-L, frozen | 0.4791 | 0.5328 | 0.5262 | **0.5394** | +0.0133 [0.002, 0.076] |
| DINOv3-H+, frozen | 0.4805 | 0.5316 | 0.5238 | 0.5380 | +0.0142 [0.003, 0.057] |
| DINOv3-B, frozen | 0.4713 | 0.5263 | 0.5241 | 0.5366 | +0.0125 [0.001, 0.059] |
| SimDINOv2-B, frozen | 0.4720 | 0.5294 | 0.5207 | 0.5362 | +0.0154 [0.005, 0.063] |
| SimDINOv2-L, frozen | 0.4756 | 0.5344 | 0.5259 | 0.5357 | +0.0098 [-0.002, 0.054] |
| DINOv3-L + LoRA r=8, seed 0 | 0.4856 | 0.5200 | 0.5218 | 0.5353 | +0.0135 [-0.001, 0.062] |
| SimDINOv2-B + LoRA r8a16, seed 0 | 0.4748 | 0.5184 | 0.5217 | 0.5306 | +0.0089 [-0.005, 0.025] |
| SimDINOv2-L + LoRA r=8, seed 0 | 0.4808 | 0.5191 | 0.5254 | 0.5349 | +0.0094 [0.003, 0.055] |

- Mean-field smoothing helps on single tiles for all eight backbones (+0.034 to +0.059 F1, every CI excluding zero).
- Adding smoothing on top of the 3x3 average gives a smaller extra boost (+0.009 to +0.015), with CIs excluding zero on five of the eight backbones.
- After context and smoothing, all eight backbones fall in a narrow range from 0.5306 to 0.5394 (a 0.009 spread), compared to 0.4713 to 0.4856 on single tiles (0.014). The gap between models shrinks after post-processing, though I did not test that difference formally.
- The highest point estimate is frozen DINOv3-L at 0.5394, just 0.0014 ahead of DINOv3-H+. No paired tests were run between backbones here, so this is just an ordering of point estimates. The SimDINOv2-B reference is 0.5362 here vs 0.5359 in its own run.
- On the 3x3 setup, five of the eight models pick interior grid values, while DINOv3-H+ and DINOv3-L still hit the maximum number of iterations (12) and SimDINOv2-B LoRA hits the maximum beta (16). Several single-tile rows also hit a grid boundary, so smoothing gains may still be slightly under-estimated.
- The three fine-tuned models are single-seed (seed 0) here, and frozen models have only one run, so none of these rows has an inter-seed standard deviation.

### Does PEFT help under context and smoothing?

Each fine-tuned model above has a matching frozen backbone. Here are the paired differences over the same 9 test orthomosaics:

| Pair | Single tile, PEFT minus frozen | 3x3 + mean field, PEFT minus frozen | Difference in smoothing gain |
|---|---|---|---|
| DINOv3-L | +0.0064 [-0.041, +0.020] | -0.0042 [-0.108, +0.010] | +0.0002 [-0.058, +0.012] |
| SimDINOv2-B | +0.0028 [-0.011, +0.018] | -0.0056 [-0.051, +0.010] | -0.0065 [-0.052, +0.0003] |
| SimDINOv2-L | +0.0053 [-0.009, +0.013] | -0.0009 [-0.012, +0.016] | -0.0004 [-0.008, +0.041] |

- On single tiles, PEFT is slightly ahead in all three pairs (+0.003 to +0.006), which is consistent with the 42-model leaderboard.
- Once context and smoothing are applied, that small advantage disappears and slightly flips (-0.006 to 0.000). None of the three differences is statistically different from zero with 9 maps.
- PEFT does not get a bigger boost from smoothing either (-0.007 to +0.000). In short, there is no evidence that fine-tuning helps once the neighbourhood is used; the direction is consistent across all three pairs, even if 9 maps are not enough to prove it strictly.

## Learned aggregation

![Learned aggregation](figures/readme/05_learned_aggregation.png)

- Self-attention over the 3x3 window gets 0.5085 (seeds 0.4928 to 0.5204), below the simple 3x3 average at 0.5205 (-0.012, CI [-0.044, +0.002]).
- Shuffling the neighbour positions inside the 3x3 window leaves attention at 0.5115 vs 0.5085 (paired delta -0.0020, CI [-0.006, +0.005], p = 0.36), so there is no evidence that the attention layer actually exploits spatial layout.
- On the larger graph that also includes unannotated windows, GATv2 is not significantly different from a uniform graph mean: -0.0033 with 2 layers (p = 0.64) and +0.0009 with 3 layers (p = 0.82), while the CIs still allow GATv2 to be roughly 0.05 worse.
- GATv2 does rely on the real graph structure: randomly rewiring the edges drops F1 by 0.024 (CI [+0.014, +0.033]).
- Notice also that the uniform mean on the graph with unannotated windows scores 0.028 lower than the annotated-only 3x3 mean. Because both the input graph and the aggregation differ between those runs, this does not isolate a single cause.

## Embedding geometry

![Embedding geometry](figures/readme/06_latent_geometry.png)

Left panel: across the 35 models in `results/rapport_data/loo_triage.csv`, how often each geometry metric orders a pair of models the same way as their test F1, with the sign and linear fit learned on the other 34 models (leave-one-model-out). These values were recomputed on the clean-baseline F1 scores (1,190 pairs); the earlier version based on registry F1 is preserved in `loo_triage_registry_f1.*`.

- Class-separation / cluster metrics do best: Calinski-Harabasz 69.1%, NC1 68.7%, Fisher ratio 68.6%, Davies-Bouldin 67.1% (grouped with neighbourhood metrics in the plot), and silhouette 61.8% (chance is 50%).
- Spectral metrics are weaker: stable rank 64.0%, participation ratio 63.5%, RankMe 56.6%, effective rank 56.3% (RankMe measures the effective rank of the embedding matrix). The two LogME transferability scores fall below chance under leave-one-model-out: 44.4% on train (p_BH = 0.16, not significant) and 42.7% on test (p_BH = 0.046), because the sign fitted on the remaining models does not generalize (rho_pred = -0.39 and -0.31).
- In practice, these metrics can roughly rank models, but they do not predict F1 values: the error reduction over just predicting the mean F1 is at most around 0.0001 on a tier whose F1 range is about 0.04.
- Because model pairs share models, they are not independent. The p-values come from a per-model Wilcoxon test with Benjamini-Hochberg correction inside each metric family, not from the raw pair count.

Middle panel (exploratory): where frozen SMARTIES-L (ViT-L, mean pooling, tested with both its Maxar-RGB and Sentinel-2 projections) ranks among the 11 frozen models on six geometry scores (1 = lowest value), compared to its F1 rank. SMARTIES is a single extraction outside the clean baseline and outside the 35 models on the left, so it is overlaid separately and does not enter any of the percentages above.

Right panel: keeping only the first 100 PCA components retains 97.4% to 100.3% of the full-embedding F1 across 5 backbones. The two exploratory SMARTIES-L curves (dashed), evaluated with the same script, recover a bit less: 96.6% (Maxar-RGB) and 93.7% (Sentinel-2) at 100 components, and 98.3% / 95.3% at 200 components.

With its Maxar-RGB projection, SMARTIES-L has the lowest RankMe (17.5, whereas the next lowest is Scale-MAE at 79.5), the lowest stable rank (1.8), the highest anisotropy (0.99), and the lowest k-NN purity (0.58) of the 12 models, yet the highest Fisher ratio (0.34). Its F1 is 0.4206 (between SatMAE and Scale-MAE), while the Sentinel-2 projection gets 0.3812 (below every frozen model). Four of the six geometry ranks are identical between the two projections even though their F1 differs by 0.039, so those scores did not pick up that gap. Numbers are in `results/rapport_data/geometry_smarties.csv`, `smarties_probe_f1.csv`, and `dim_probe_smarties_*.json`.

## Data size

![Data curve](figures/readme/07_data_curve.png)

- On random subsets, LoRA stays above the frozen probe at every training fraction (+0.005 to +0.023). With only 10% of the training tiles, LoRA reaches 0.4725, roughly matching the frozen probe trained on 100% of the tiles (0.4712).
- On spatially blocked subsets, performance drops sharply compared to random sampling: 0.099 vs 0.426 at 1%, and 0.434 vs 0.480 at 50%. At 1%, spatially blocked subsets only contain 3 to 6 of the 8 main classes. Note that missing classes and spatial shift are mixed together here and were not separated.

## Per class

![Per-class F1 and class support](figures/readme/10_per_class_f1.png)

| Class | Train tiles | Test tiles | Frozen DINOv3-H+ | Context fusion (fine-tuned) |
|---|---:|---:|---:|---:|
| ALDE | 5,285 | 1,884 | 0.958 | 0.962 |
| ARCA | 88 | 113 | 0.000 | 0.000 |
| BIRC | 12,845 | 4,937 | 0.837 | 0.867 |
| DRYI | 390 | 177 | 0.000 | 0.000 |
| LICH | 5,334 | 1,615 | 0.635 | 0.663 |
| MOSS | 5,089 | 1,935 | 0.520 | 0.600 |
| PETF | 1,990 | 1,037 | 0.506 | 0.572 |
| RUBC | 330 | 31 | 0.000 | 0.000 |
| SEDG | 5,250 | 2,134 | 0.601 | 0.694 |
| TUSS | 8,306 | 3,189 | 0.742 | 0.769 |
| WILL | 4,374 | 546 | 0.486 | 0.460 |

Three classes get an F1 of exactly zero for both models (across all 3 seeds for context fusion, and in the single run for frozen DINOv3-H+). Across all 230 probes in the clean baseline, the highest scores ever reached on those classes are 0.082 (ARCA), 0.026 (DRYI), and 0.146 (RUBC). Because these are the rarest classes, they pull the 11-class macro F1 down for every model, so it makes more sense to compare models relative to each other than to judge the absolute macro F1 alone. A twelfth class (RHOL) does not appear in validation or test and is excluded.

### The same ranking on the 8 remaining classes

![8-class view](figures/readme/08_classes_8.png)

The scatter plot shows the 42 configurations along with the probe-arms, decoding, and aggregation setups as red diamonds (each from its own run). The rank-change bar chart on the right covers only the 42 configurations.

Since ARCA, DRYI, and RUBC stay close to zero for all models, they lower macro F1 without really helping separate models. The 8-class view leaves them out of the macro average (321 out of 17,598 test tiles) while keeping the exact same 11-class classifier: their tiles remain in the test set, so a kept class predicted on those tiles still counts as a false positive, and a kept tile predicted as one of those three still counts as a false negative. Those three classes are simply excluded from the mean. This matches the `f1_macro_8cls_test` convention logged by training runs, and is not a classifier retrained on 8 classes (earlier 8-class refits used a different protocol and are not used here). Note also that `f1_8cls` in `recipe_comparisons.json` and `gnn_canonical.json` removes the 321 tiles before scoring, which is roughly +0.006 higher; the table and figure below use the first convention.

- Dropping the three rare classes raises macro F1 by about 0.18 (median +0.179): the 42 configurations go from 0.3547–0.5079 (11 classes) to 0.4805–0.6984 (8 classes).
- The model ordering almost does not change: Spearman 0.991, Kendall 0.965. 26 models keep the exact same rank, 10 move by 1 place, 4 by 2 places, and only 2 move by 3 or more.
- Those two models are DINOv3-B with full fine-tuning (9 places down) and DINOv3-B with attention-only tuning (3 places down). That shift comes entirely from the three dropped classes, which account for 3/11 of the 11-class average: DINOv3-B full has the highest ARCA F1 among the 42 models (0.039, worth about +0.0035 in macro F1). Only nine models get a non-zero ARCA score, and none exceeds 0.040.
- The best recipe (3x3 mean + mean field) gets 0.5359 in 11 classes and 0.7369 in 8 classes.

Rows are sorted by ascending 8-class F1; the last column indicates which run each value comes from.

| Arm | 11 classes | 8 classes | Run |
|---|---:|---:|---|
| GATv2, edges shuffled | 0.4648 | 0.6391 | aggregation |
| Single tile | 0.4719 | 0.6489 | decoding |
| CNN on the 3x3 window | 0.4824 | 0.6575 | probe-arms |
| GATv2, 2 layers | 0.4891 | 0.6725 | aggregation |
| Uniform mean over the graph | 0.4924 | 0.6770 | aggregation |
| GATv2, 3 layers | 0.4933 | 0.6783 | aggregation |
| Attention over the 3x3 window | 0.5085 | 0.6957 | probe-arms |
| Attention, positions shuffled | 0.5115 | 0.6999 | probe-arms |
| 3x3 mean | 0.5207 | 0.7160 | decoding |
| Single tile + mean field | 0.5297 | 0.7284 | decoding |
| 3x3 mean + ICM | 0.5332 | 0.7331 | decoding |
| 3x3 mean + mean field | 0.5359 | 0.7369 | decoding |

These setups correspond to the red diamonds in the figure above. Single tile is 0.6488 in the 42-configuration table (clean baseline) and 0.6489 here (decoding run).

## Protocol

| Item | Value |
|---|---|
| Data | Arctic-TVC, drone orthomosaics at about 2.2 mm/px, Trail Valley Creek, NWT. [doi:10.20383/103.01418](https://doi.org/10.20383/103.01418) |
| Split | By whole orthomosaic (no tile leakage): 15 train, 8 val, 9 test (49,281 / 13,209 / 17,598 tiles) |
| Classes | 11 (RHOL removed). Metric: macro F1 over the 11 classes |
| Probe | Multinomial logistic regression (lbfgs), C in {1e-4, 1e-3, 0.01, 0.1, 1, 10} selected on val |
| Intervals | Paired hierarchical bootstrap (orthomosaic then tile), 10,000 replicates, Benjamini-Hochberg within a pre-declared family |
| Determinism | seed 42, single-thread BLAS (changing the BLAS thread count alone shifts F1 by about 0.0015) |

## Ongoing work

The follow-up work currently in progress focuses on whether these findings generalize beyond Arctic-TVC:

- **A second dataset (Léo).** Four aerial sites in southern Québec, about 355k nadir tiles at 224 px. I use it as an external validation dataset, not as extra training data for the benchmark above. No results from it are reported in this README yet.
- **Self-supervised pretraining on Léo.** `leo-explora-ssl/` adapts a SimDINOv2-B/16 checkpoint on Léo tiles using an ExPLoRA-style recipe (LoRA on early blocks, DINO + iBOT + KoLeo, EMA teacher, 3 seeds) and probes the resulting checkpoints back on Arctic-TVC. So far the result is negative: across all 10 checkpoints (epochs 5 to 50, mean of 3 seeds), the adapted features stay below frozen SimDINOv2-B in both the 3x3 context view (-0.002 to -0.013 vs 0.4932) and the fused view (-0.007 to -0.017 vs 0.5059). On single tiles, it only catches up to parity by epoch 50 (+0.0005 vs 0.4721). Since the frozen baseline is a single deterministic run, there is no paired CI here and the consistent 10-out-of-10 pattern is the main evidence. A 50-epoch run may still be under-trained, so a longer run is not ruled out. Results: `results/leo_explora_ssl/`.
- **Domain shift.** A first pilot found a correlation between distribution discrepancy (MMD, optimal transport) and transfer performance, and that association replicated in a second experiment. However, I could not show that discrepancy adds information beyond RankMe, so I do not make any claim about it yet. Discrepancy-guided alignment is on hold until a discrepancy score shows clear incremental value over RankMe.
- **Broader model screening.** Initial frozen probes for three extra foundation models all score below DINOv3-H+ (0.4805): SigLIP 2 ViT-B/16 at 0.4475, SatlasPretrain Aerial SwinB at 0.4373, and C-RADIO v4-H at 0.3764. Not all three are Earth-observation models, they are single-seed and outside the canonical ranking, and their preprocessing is not yet fully aligned (ImageNet normalization assumed for Satlas; C-RADIO evaluated at 224 px instead of its native 512 px), so this is not a fair head-to-head comparison yet.

The domain-shift reports (P0 pilot and P1 replication) are kept locally only. All scalar outputs supporting the sections above are committed in this repository.

## Limits

- Having only 9 test orthomosaics limits statistical power. Confidence intervals are wide, and percentile intervals are not symmetric around the point estimate (for instance +0.0152 with CI [+0.006, +0.062]). They should be read mainly as "excludes zero or not" rather than as tight error bars. I did not run equivalence tests: with 9 maps, this setup can reliably detect differences of about 0.03 to 0.05, while smaller gaps remain unresolved rather than proven equal. For a few pairs, the CI and p-value disagree (e.g., fine-tuned ResNet-50 ImageNet vs best frozen: CI [-0.061, -0.015], p = 0.094, q = 0.22); following the protocol, I report the Benjamini-Hochberg decision, so that pair does not pass.
- Scores in the ranking, adaptation, and geometry sections come from the clean baseline (2026-10-08); the earlier registry-F1 geometry file is kept as `loo_triage_registry_f1.*`. The context, smoothing, and aggregation results come from their own runs: frozen single-tile SimDINOv2-B/16 varies slightly between 0.4718 and 0.4720 across runs, and the 3x3 mean is 0.5205 or 0.5207. Every paired comparison is computed within the same run.
- Frozen backbones are deterministic single runs, so seed SD is not defined for them.
- Context and smoothing are tested across eight backbones (five frozen, three LoRA at seed 0). GATv2 and the other learned aggregation variants are tested on SimDINOv2-B/16 only.
- Invalid ExPLoRA runs and deprecated single-seed fine-tunes (`resnet50_arctic`, `vitb16_arctic`, `vitb16_fulft_arctic`) are excluded. The `_old` keys still present in the baseline are 3-seed SOTA-screening runs, not those deprecated ones. Note that `tests/test_repo_guards.py` checks the LaTeX manuscript for retired values, not this README.

## Reproduce

Cloning this repository gives you all the code, configs, 11-class splits, and scalar tables needed to verify every number above. The raw orthomosaics (public on FRDR), 224 px tile images, extracted embeddings, and model weights (some gated) are not included.

From a fresh clone, you can rebuild all ten figures directly from the committed tables:

```bash
pip install -r requirements.txt          # or the pinned versions in requirements-lock.txt
python scripts/make_readme_plots.py      # rebuilds all ten figures from results/
```

Running the full pipeline from scratch requires downloading Arctic-TVC, tiling the orthomosaics, and a GPU for fine-tuning:

```bash
# 12-class split CSVs are committed; this regenerates the 11-class splits
python scripts/generate_splits_11cls.py --splits-dir splits --out-dir splits --force

# feature extraction and canonical linear probe
python extract.py --config configs/frozen_dinov3_lvd.yaml
python probe.py   --config configs/frozen_eval.yaml
python analyze.py --config configs/frozen_eval.yaml
```

Two practical notes:

- The committed splits are the canonical ones, generated from `splits/split_v3_assignment.json` (15 / 8 / 9 orthomosaics, verified against the CSVs). Both 12-class and 11-class CSVs are included, and the command above reproduces the 11-class CSVs byte-for-byte from the 12-class ones. The raw raster tiling scripts are not published in this repo.
- Fine-tuning requires GPUs and access to gated weights. Per-seed fine-tuned scores are committed in `results/baseline_clean_2026-10-08/per_seed.csv`, so you do not need a GPU to verify the tables or regenerate the figures.

## Where the numbers come from

| Section | Values | File |
|---|---|---|
| Ranking, adaptation, per class, 8-class view | per-seed and per-class F1 | `results/baseline_clean_2026-10-08/per_seed.csv`, `per_class_f1.csv` |
| Significance of the ranking | paired differences, CIs, q-values | `results/baseline_clean_2026-10-08/statistics/pairwise_effects.csv`; predictions and protocol in `preds/` and `statistics/STATISTICAL_PROTOCOL.md` |
| Context, smoothing, learned aggregation | F1, deltas, CIs | `results/tile_context_2026-11-01/*.json` |
| Embedding geometry | pair ordering, leave-one-model-out | `results/rapport_data/loo_triage.csv`, `dim_probe_summary.csv` |
| Data size | learning curves, blocked subsets | `results/datacurve_lora/*_agg.csv`, `lora_spatial_v2/results_spatial_summary.csv`, `spatial_datacurve/class_composition.csv` |
| Per-class table | tile counts, per-class F1 | `splits/*_11cls.csv`, `per_class_f1.csv` |
| Non-linear readouts | 0 of 33, 2 of 24 | `results/rapport_data/head_sweep_summary.csv` |
| Context and aggregation arms, 8 classes | 12 arms, two schemas | `results/tile_context_2026-11-01/arms_8cls.csv` |
| Context and smoothing on 8 backbones | F1, paired deltas | `results/tile_context_2026-11-01/multibackbone_smoothing.csv` |
| PEFT against its frozen backbone | paired differences | `results/tile_context_2026-11-01/peft_pairs.csv` |
| Deployable or upper bound | flag per configuration | `results/baseline_clean_2026-10-08/summary.csv` |

Two 8-class scoring conventions appear in the committed files: `arms_8cls.csv` and `per_class_f1.csv` keep the 321 test tiles from the three dropped classes (matching `f1_macro_8cls_test`), whereas `f1_8cls` in `recipe_comparisons.json` and `gnn_canonical.json` removes those tiles before scoring (about +0.006 higher). The README table and figures use the first convention.

Repository layout: `src/` core package, `extract.py` / `probe.py` / `train.py` / `analyze.py` entry points, `configs/` YAML configs, `scripts/` analysis and SLURM scripts, `research/gnn_grid_2026-10-31/code/` tile context, smoothing and GNN code, and `results/` scalar outputs.

Arctic-TVC is publicly available on FRDR under its own licence ([doi:10.20383/103.01418](https://doi.org/10.20383/103.01418)). Code (Python, YAML, Makefile, CI) is under the [MIT License](LICENSE); figures and result tables (PNG, PDF, JSON, CSV) are under [CC-BY-4.0](LICENSE-DATA). Citation metadata is in [CITATION.cff](CITATION.cff).
