# Contexte spatial déployable sur les 8 vraies voisines (26–27 nov. 2026)

Réplication complète de la table 8 backbones (`multibackbone_smoothing_fullgraph.csv`) et des paires PEFT (`peft_pairs_fullgraph.csv`) sur le graphe complet à 8,00 vraies voisines par tuile (80 088 tuiles annotées + 40 076 tuiles auxiliaires hors transect = 120 164 nœuds, sans masque d'annotation).

## Provenance Narval (vérifiée)
- Extraction des tuiles auxiliaires pour les 8 backbones : job GPU `10765124` (`COMPLETED 0:0`, 31 min 09 s, alignement vérifié $\cos \ge 0{,}99989$ contre `features_tile.npy`).
- Sondes, lissage (uniforme, ICM, pondération par confiance `Wconf` $\gamma=4$) et bootstrap hiérarchique sur 10 000 réplicats : job CPU `10765125` (`COMPLETED 0:0`, 31 min 23 s, BLAS mono-thread).

## Fichiers de sortie
- `out/multibackbone_smoothing_fullgraph.csv` : 8 backbones (`R0`, `R0+MF`, `R0+Wconf`, `R3`, `R3+MF`, `R3+Wconf`, IC95 hiérarchiques).
- `out/peft_pairs_fullgraph.csv` : 3 paires PEFT contre gelé (`DINOv3-L`, `SimDINOv2-B`, `SimDINOv2-L`).

## Résumé des constats
1. Le meilleur score déployable atteint 0,5089 (11 cls) et 0,6998 (8 cls) pour `DINOv3-B` gelé `R3 + Wconf`, devant `SimDINOv2-L` (0,5076), `DINOv3-L` (0,5065) et `DINOv3-H+` `R0 + Wconf` (0,5053), au même plafond que le zoom-out concaténé `R2` (`0,5059`).
2. Le gain de lissage reste positif sur les 8 backbones : `R0 + MF` ajoute `+0,013` à `+0,023` ($p \le 0{,}015$) et `R0 + Wconf` ajoute `+0,012` à `+0,031` ($p \le 0{,}010$), avec tous les IC95 au-dessus de 0.
3. L'avantage du PEFT sur tuile seule disparaît et s'inverse légèrement sur les 3 paires sous `R0 + Wconf` (`-0,0034` à `-0,0086`).
4. Restreindre le voisinage aux 6,53 voisines annotées (« bons voisins ») montait artificiellement le score à `0,5306–0,5394` (+0,025 à +0,035), et aucune pondération ni aucun classifieur (`W_domain` AUC 0,642 → 0,4967 ; `W_appris` AUC 0,792 → 0,4975 ; `W_conf` 0,4978) ne permet de séparer les bons des mauvais voisins depuis les images (cf. `research/gnn_grid_2026-10-31/out/decoding_full/weighted_fullgraph.json`).
