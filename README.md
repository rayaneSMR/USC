# UWOC-SemCom

Implémentation PyTorch du pipeline de communication sémantique numérique pour
canal optique sans fil sous-marin (Underwater Wireless Optical Communication —
UWOC), avec quantification FSQ, canal IM/DD différentiable, débruiteur latent
IRSNet, et entraînement multi-phases alterné (AMP).

## Structure du repo

- `net/` — tous les modules du modèle (backbone, bottleneck FSQ, canal UWOC,
  IRSNet, générateur, têtes de tâches, pipeline complet).
- `datasets/` — chargement des données (images coraux + masques).
- `losses/` — fonctions de perte par phase.
- `engine/` — boucles d'entraînement par phase + orchestrateur AMP.
- `utils/` — logger, checkpoints, gel/dégel de modules, métriques.
- `evaluation/` — courbes BER, fit du modèle de perte sémantique ξ(ρ,γ).
- `configs_yaml/` — configurations de codebook FSQ (16/14/10/5 bits) et canal.

## Démarrage rapide

```bash
pip install -r requirements.txt

# Phase 1 : extraction de caractéristiques (sans bruit)
python main.py --phase 1 --config configs_yaml/codebook_14bit.yaml --mode train

# Phase 2 : renforcement de robustesse (BFP + MATK)
python main.py --phase 2 --config configs_yaml/codebook_14bit.yaml --mode train \
    --resume checkpoints/phase1_last.pt

# Phase 3 : alignement entraînement-test
python main.py --phase 3 --config configs_yaml/codebook_14bit.yaml --mode train \
    --resume checkpoints/phase2_last.pt

# Phase 4 : adaptation des têtes de tâches
python main.py --phase 4 --config configs_yaml/codebook_14bit.yaml --mode train \
    --resume checkpoints/phase3_last.pt

# Évaluation complète
python main.py --phase 4 --config configs_yaml/codebook_14bit.yaml --mode eval \
    --resume checkpoints/phase4_last.pt
```

## Référence

Voir `Pipeline.pdf` pour la description scientifique complète (contexte,
formulation du canal, protocole d'entraînement en 4 phases, modèle de perte
sémantique paramétrique).
