# Classical colour and texture descriptors

Scripts for testing whether concatenating classical radiomics / texture descriptors to frozen ViT embeddings improves tundra vegetation classification on Arctic-TVC (~2.2 mm/px).

- Module: `src/texture.py` (pure NumPy/SciPy + PyWavelets, no C build required)
- Extraction: `scripts/texture_features.py`
- Probe ablation & paired bootstrap: `scripts/texture_ablation.py`
- Redundancy analysis: `scripts/texture_redundancy.py`
- Unit tests: `tests/test_texture_features.py`

## Feature families (106 features by default)

| Family | Count | Captures | Rotation invariant |
|---|---:|---|---|
| `glcm` | 30 | Pixel-pair transitions and contrast (Haralick 1973) | No (averaged across 4 directions) |
| `glrlm` | 16 | Run-length linear structures (Galloway 1975) | No |
| `glszm` | 16 | Connected same-level zone sizes (Thibault 2009) | Yes |
| `gldm` | 15 | Local grey-level dependence (Sun & Wee 1983) | Yes |
| `ngtdm` | 5 | Coarseness and busyness vs local neighbourhood (Amadasun & King 1989) | Yes |
| `dwt` | 10 | Wavelet sub-band energies across scale and orientation | Yes |
| `fourier` | 14 | Power spectrum slope and radial/angular energy | Yes |

Descriptors are concatenated to the frozen tile embedding before the linear probe (never stacked as raw input channels into a pretrained ViT).

## Usage

```bash
# Requires only numpy, scipy, PyWavelets (and scikit-learn for probing)
python tests/test_texture_features.py
python scripts/texture_features.py --tiles-dir out/tiles --splits-dir splits --out-dir results/texture
python scripts/texture_ablation.py --help
```

## Result summary

On Arctic-TVC (11 classes), concatenating classical colour and texture features to frozen DINOv3-B or SimDINOv2-B embeddings reduces test macro F1 by 0.004 to 0.005 (results in `results/texture/`).
