# UWOC-SemCom

Pipeline de communication sémantique numérique de bout en bout pour canal optique sans fil sous-marin (**Underwater Wireless Optical Communication — UWOC**), implémenté en **PyTorch**.

Le pipeline intègre une quantification scalaire finie (**FSQ**), une modélisation physique réaliste du canal optique **IM/DD** (atténuation Beer-Lambert, turbulence avec moyennage d'ouverture, erreurs de pointage dynamiques et bruit SiPM/TIA), un débruiteur latent multi-échelle **IRSNet**, ainsi qu'un protocole d'entraînement multi-phases alterné (**AMP**).

---

## 1. Vue d'ensemble de l'architecture

Le système suit une chaîne de transmission sémantique complète :

```
Image x (3×H×W)
   │
   ▼
[ÉMETTEUR]   Backbone → Fusion multi-échelle ASFF → Goulot Edge (SE+Conv1x1) → FSQ → Mappage binaire
   │
   ▼  flux binaire b ∈ {0, 1}
[CANAL]      Modulation IM/DD NRZ-OOK → Gain h = h_c · h_t · h_p → Bruit SiPM/TIA → Seuil optimal γ_th
   │
   ▼  bits reçus b̂ (BER p via erfc)
[RÉCEPTEUR]  Démappage binaire → Écrêtage latent → IRSNet → Générateur G_ψ → Image reconstruite x̂
                                                                               │
                                                                   ┌───────────┴───────────┐
                                                                   ▼                       ▼
                                                            Segmentation (Dice/CE)   Classification (CE)
```

### Modules clés

1. **Émetteur** :
   * **Backbone** (`net/backbone.py`) : `simple_cnn` (développement léger), `yolo26` (features P3/P4/P5 via hooks) ou `segformer_mitb5` (features multi-échelles 4 niveaux).
   * **Fusion multi-échelle ASFF** (`net/multiscale_fusion.py`) : fusion pondérée adaptative sans concaténation mémoire-lourde.
   * **Goulot d'étranglement Edge** (`net/edge_bottleneck.py`) : attention Squeeze-and-Excitation (SE) + convolution $1\times 1$ projetant vers $d \le 8$ canaux.
   * **Quantification FSQ + STE** (`net/fsq.py`) : quantification sans codebook explicite, différentiable via Straight-Through Estimator.
   * **Mappage Binaire** (`net/bit_mapping.py`) : conversion FSQ $\leftrightarrow$ flux binaire OOK avec décomposition division/modulo compatible ONNX.

2. **Canal Physique Optique Sous-Marin** (`net/channel.py`) :
   * **Atténuation Beer-Lambert** : $h_c = \exp(-c_e L)$ avec $c_e \approx 0.151\,\text{m}^{-1}$ (eau claire).
   * **Turbulence océanique log-normale** : régime de faible turbulence avec **moyennage d'ouverture** (lentille $D_r = 75$ mm), $\sigma_T^2 \approx 1.1 \times 10^{-7}$ (calibré selon l'Équation 43 de l'article).
   * **Erreurs de pointage géométriques (Équation 24)** : calcul dynamique du rayon de faisceau $w_L(L) = \sqrt{w_0^2 + (\theta_{\text{div}} L)^2}$ et du faisceau équivalent :
     $$\nu = \frac{D_r \sqrt{\pi}}{2\sqrt{2} w_L(L)}, \quad A_0 = (\text{erf}(\nu))^2, \quad w_{Leq}^2 = w_L(L)^2 \frac{\sqrt{\pi} \text{erf}(\nu)}{2\nu\exp(-\nu^2)}$$
     $$h_p = A_0 \exp\left(-\frac{2 r_d^2}{w_{Leq}^2}\right), \quad r_d = \sigma_{\theta t} L |N(0, 1)|$$
   * **Bruit SiPM/TIA post-détection** : bruit de grenaille dépendant du signal + bruit thermique Johnson-Nyquist.
   * **Bande passante LPF électrique liée au débit** : $B_e \approx R_b / 2$ ($0.5$ MHz pour $R_b = 1$ Mbps).
   * **Seuil de détection optimal et BER exact** : seuil gaussien $\gamma_{th}$ et calcul analytique exact de la probabilité d'erreur binaire par pixel via $\text{erfc}$.

3. **Récepteur** :
   * **Écrêtage latent** (`net/latent_clamp.py`) : restriction stricte aux bornes FSQ théoriques $[-(L_i-1)/2, +(L_i-1)/2]$.
   * **IRSNet** (`net/irsnet.py`) : U-Net résiduel multi-échelle restaurant le tenseur latent bruité par le canal.
   * **Générateur visuel $G_\psi$** (`net/generator.py`) : décodeur à convolutions transposées avec sortie Sigmoid $\hat{x} \in [0, 1]^{3\times H\times W}$.
   * **Têtes aval** (`net/heads.py`) : segmentation multi-classes (CoralMask) et classification de santé corallienne.

4. **Échantillonneur Stratifié (Stratified Sampler)** (`datasets/sampler.py`) :
   * Résout l'instabilité du gradient en Phase 4 sur les jeux de données mixtes (segmentation + classification pure).
   * Garantit qu'**au moins 50% de chaque batch** contient des images avec de vrais masques de segmentation (`has_mask=True`), assurant un gradient Dice non nul à chaque itération.

---

## 2. Structure du Dépôt

```
uwoc-semcom/
├── args.py                 # Définition des arguments de la ligne de commande (argparse)
├── config.py               # Configuration globale par défaut et hyperparamètres physiques
├── main.py                 # Point d'entrée principal pour l'entraînement et l'évaluation
├── requirements.txt        # Dépendances Python
│
├── configs_yaml/           # Fichiers de configuration YAML
│   ├── codebook_5bit.yaml          # FSQ 5 bits/token (L=[2,2,2,2,2], d=5) — haute résilience
│   ├── codebook_10bit.yaml         # FSQ 10 bits/token (L=[8,5,5,5], d=4) — forte compression
│   ├── codebook_14bit.yaml         # FSQ 14 bits/token (L=[8,8,8,6,5], d=5) — nominal
│   ├── codebook_16bit.yaml         # FSQ 16 bits/token (L=[8,8,8,5,5,5], d=6) — haute fidélité
│   └── channel_shallow_clear.yaml  # Paramètres physiques du canal UWOC (Ijeh et al.)
│
├── datasets/               # Gestion des jeux de données
│   ├── coral_dataset.py    # CoralDataset (réel) & DummyCoralDataset (synthétique)
│   └── sampler.py          # StratifiedSampler & StratifiedBatchSampler
│
├── net/                    # Architecture des réseaux neuronaux
│   ├── backbone.py         # SimpleCNN, YOLO26, SegFormer MiT-B5
│   ├── multiscale_fusion.py# ASFF (Adaptively Spatial Feature Fusion)
│   ├── edge_bottleneck.py  # Squeeze-and-Excitation + projection 1x1
│   ├── fsq.py              # Finite Scalar Quantization + STE
│   ├── bit_mapping.py      # Mappage entier FSQ <-> flux binaire
│   ├── channel.py          # Canal physique IM/DD NRZ-OOK différentiable
│   ├── latent_clamp.py     # Écrêtage latent
│   ├── irsnet.py           # Dénoyeur latent résiduel multi-échelle
│   ├── generator.py        # Décodeur visuel RGB
│   ├── heads.py            # Têtes aval de segmentation et classification
│   └── pipeline.py         # Orchestrateur UWOCSemComPipeline complet
│
├── engine/                 # Boucles d'entraînement multi-phases (AMP)
│   ├── amp_scheduler.py    # Planificateur alterné multi-phases
│   ├── phase1_fe.py        # Phase 1 : Extraction de caractéristiques (régime parfait)
│   ├── phase2_re.py        # Phase 2 : Renforcement de robustesse (bruit/MATK)
│   ├── phase3_tta.py       # Phase 3 : Alignement entraînement-test (canal physique)
│   └── phase4_task.py      # Phase 4 : Adaptation des têtes aval
│
├── losses/                 # Fonctions de perte
│   ├── reconstruction.py   # Pertes L1 / MSE visuelles
│   ├── latent_align.py     # Alignement latent + calculs multi-phases (L_P1, L_P2, L_P4)
│   └── task_losses.py      # Cross-Entropy + Dice Loss
│
├── utils/                  # Utilitaires (checkpoints, compression, métriques)
└── tests/                  # Suite de tests unitaires et d'intégration
```

---

## 3. Installation

Nécessite Python 3.10+ et PyTorch 2.0+.

```bash
# Cloner le dépôt
git clone https://github.com/MohamedNennouche/task-oriented-UWOC.git
cd task-oriented-UWOC

# Installer les dépendances
pip install -r requirements.txt
```

---

## 4. Utilisation & Entraînement

### Lancement avec configurations YAML

Le script `main.py` parse dynamiquement les fichiers YAML passés via `--config` et écrase les paramètres de configuration correspondants (notamment les niveaux FSQ `levels`, la dimension latente $d$, la distance du canal et les hyperparamètres d'entraînement) :

```bash
# Entraînement complet (planificateur AMP 10 époques) avec configuration nominale 14 bits
python main.py --config configs_yaml/codebook_14bit.yaml

# Test avec configuration 5 bits (haute résilience aux faibles SNR)
python main.py --config configs_yaml/codebook_5bit.yaml

# Test avec configuration 10 bits (forte compression)
python main.py --config configs_yaml/codebook_10bit.yaml

# Test avec configuration 16 bits (haute fidélité optique)
python main.py --config configs_yaml/codebook_16bit.yaml
```

### Entraînement ciblé par phase (AMP)

Vous pouvez cibler une phase spécifique grâce à `--phase` :

```bash
# Phase 1 (FE) : Reconstruction d'image sans bruit de canal
python main.py --config configs_yaml/codebook_14bit.yaml --phase 1 --epochs 5

# Phase 2 (RE) : Entraînement du débruiteur IRSNet et de l'alignement latent
python main.py --config configs_yaml/codebook_14bit.yaml --phase 2 --epochs 5

# Phase 3 (TTA) : Alignement en présence du canal physique complet
python main.py --config configs_yaml/codebook_14bit.yaml --phase 3 --epochs 5

# Phase 4 : Adaptation des têtes de tâches (Segmentation Dice/CE + Classification CE)
python main.py --config configs_yaml/codebook_14bit.yaml --phase 4 --epochs 5
```

### Surcharges en ligne de commande

Les arguments passés en ligne de commande écrasent la configuration YAML :

```bash
python main.py --config configs_yaml/codebook_5bit.yaml \
    --link_distance 20.0 \
    --batch_size 8 \
    --lr 0.0002 \
    --data_root "chemin/vers/merged_dataset"
```

---

## 5. Suite de Tests Unitaires

Une suite de tests vérifie l'intégrité mathématique, physique et logicielle de l'implémentation :

```bash
# Vérification du canal physique (Équation 24, B_e=R_b/2, sigma_T2=1.1e-7, monotonie du BER)
python tests/test_channel.py

# Vérification du Stratified Sampler (garantie >= 50% de masques par batch)
python tests/test_sampler.py

# Vérification du chargement dynamique des configurations YAML (5-bit, 10-bit, 14-bit, 16-bit)
python tests/test_yaml_config.py

# Vérification de la quantification FSQ et du roundtrip binaire
python tests/test_fsq.py
python tests/test_bit_mapping.py

# Vérification de la fusion multi-échelle ASFF et du débruiteur IRSNet
python tests/test_multiscale_fusion.py
python tests/test_irsnet.py

# Vérification de la compatibilité has_mask et du calcul des pertes Phase 4
python tests/test_has_mask_wiring.py
```

---

## 6. Référence Scientifique

Pour les détails théoriques complets (dérivation des équations de propagation, formulations du canal UWOC composite $h = h_c \cdot h_t \cdot h_p$, modèle de bruit SiPM/TIA et protocole d'entraînement AMP en 4 phases), se référer au document de synthèse `IMPLEMENTATION.md` et à la documentation de référence du projet.
