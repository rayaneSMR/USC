# Documentation d'implémentation — UWOC-SemCom

Pipeline de communication sémantique numérique pour canal optique sans fil
sous-marin (Underwater Wireless Optical Communication), implémenté en
PyTorch. Ce document explique en détail chaque module, les choix
d'implémentation, les bugs trouvés et corrigés en cours de route, et
comment tout s'articule.

**Référence scientifique** : document `Pipeline.pdf` fourni (contexte,
formulation du canal, protocole d'entraînement en 4 phases).

**État du projet** : pipeline complet, code-testé (25 tests unitaires),
`main.py` tourne intégralement sur données synthétiques. Le seul élément
non branché est le **vrai dataset** (CoralMask fusionné) — un générateur
de données synthétique (`DummyCoralDataset`) le remplace pour valider que
toute la mécanique fonctionne.

---

## Table des matières

1. [Vue d'ensemble de l'architecture](#1-vue-densemble-de-larchitecture)
2. [Émetteur](#2-émetteur)
3. [Canal physique optique sous-marin](#3-canal-physique-optique-sous-marin)
4. [Récepteur](#4-récepteur)
5. [Orchestration du pipeline complet](#5-orchestration-du-pipeline-complet)
6. [Entraînement en 4 phases (AMP)](#6-entraînement-en-4-phases-amp)
7. [Export ONNX et déploiement edge](#7-export-onnx-et-déploiement-edge)
8. [Compression du modèle](#8-compression-du-modèle)

10. [Structure du dépôt](#10-structure-du-dépôt)
11. [Comment lancer le projet](#11-comment-lancer-le-projet)


---

## 1. Vue d'ensemble de l'architecture

Le pipeline suit exactement le synoptique du document de référence (§2) :

```
Image x (3×H×W)
   │
   ▼
[ÉMETTEUR]  Backbone → Fusion multi-échelle → Edge Bottleneck → FSQ → Bit Mapping
   │
   ▼  flux binaire b
[CANAL]     Modulation IM/DD → h = h_c·h_t·h_p → Bruit SiPM → Seuil optimal
   │
   ▼  bits corrompus b̂ (BER p)
[RÉCEPTEUR] Démappage → Écrêtage → IRSNet → Générateur → x̂
                                                │
                                    ┌───────────┼───────────┐
                                    ▼           ▼           ▼
                                  x̂ (PSNR)   Seg (mIoU)  Cls (Top-1)
```

Chaque bloc est un module PyTorch indépendant dans `net/`, assemblé par
`net/pipeline.py::UWOCSemComPipeline`.

---

## 2. Émetteur

### 2.1 Backbone — `net/backbone.py`

Extrait les caractéristiques de l'image brute. Trois options :


| Backbone          | Classe                   | Usage                                                            |
| ----------------- | ------------------------ | ---------------------------------------------------------------- |
| `simple_cnn`      | `SimpleCNNBackbone`      | CNN léger, pas de poids externes — développement rapide       |
| `yolo26`          | `YOLO26Backbone`         | Vrais poids`ultralytics.YOLO`, backbone+neck de `yolo26n-seg.pt` |
| `segformer_mitb5` | `SegFormerMiTB5Backbone` | Encodeur`transformers.SegformerModel`, config officielle MiT-B5  |

**YOLO26Backbone** utilise un `forward_pre_hook` posé sur la dernière
couche (`Segment26`, la tête de segmentation) pour intercepter les
features juste avant qu'elles n'entrent dans la tête — on récupère ainsi
le travail du backbone+neck sans la tête elle-même. Découverte empirique
importante : cette interception renvoie une **liste de 3 échelles**
(style P3/P4/P5), pas un tenseur unique :

```
feat[0]: (B,  64, H/8,  W/8)
feat[1]: (B, 128, H/16, W/16)
feat[2]: (B, 256, H/32, W/32)
```

**SegFormerMiTB5Backbone** expose `output_hidden_states=True`, qui renvoie
directement 4 cartes déjà en format spatial `(B,C,H,W)` (pas de tokens
aplatis à reprojeter, contrairement à un ViT classique) :

```
stage1: (B,  64, H/4,  W/4)
stage2: (B, 128, H/8,  W/8)
stage3: (B, 320, H/16, W/16)
stage4: (B, 512, H/32, W/32)
```

Les deux backbones exposent une méthode commune
`extract_multi_scale_features(x) -> list[Tensor]`, utilisée par la fusion
ASFF (§2.2). Un attribut `out_channels` est calculé automatiquement à la
construction (passe factice) pour dimensionner les couches suivantes sans
configuration manuelle.

**Chargement offline** : `YOLO26Backbone._resolve_weights_path()` cherche
d'abord les poids localement (répertoire courant, puis `local_weights_dir`
si fourni) avant de laisser `ultralytics` tenter un téléchargement — utile
pour un déploiement sur une sonde sans accès réseau garanti.

### 2.2 Fusion multi-échelle ASFF — `net/multiscale_fusion.py`

Le backbone produit plusieurs échelles, mais le goulot d'étranglement Edge
attend un seul tenseur. Naïvement, on pourrait ne garder que la dernière
échelle (la plus profonde) — c'est le comportement de repli si la fusion
est désactivée — mais on perd le détail spatial fin des échelles peu
profondes.

**Choix retenu : ASFF** (*Adaptively Spatial Feature Fusion*, Liu, Huang,
Wang, 2019), pensé dès le départ pour respecter une contrainte **edge**
(la sonde sous-marine a des ressources de calcul/mémoire limitées) :

```
pour chaque échelle Cᵢ (i=1..N) :
    proj_i  = Conv1x1(Cᵢ)                    # projection vers C_ref canaux
    up_i    = Upsample(proj_i, taille_ref)   # résolution commune

    w_i     = Conv3x3(up_i) → 1 canal        # carte de poids (PAS les features)

poids = softmax([w_1, ..., w_N], dim=échelle)
fusion = Σ poids_i · up_i                     # somme pondérée, pas de concat
```

Pourquoi ce choix plutôt que BiFPN/PANet/cross-attention classiques :
ceux-ci **concatènent** les features complètes de chaque échelle avant de
les traiter, ce qui multiplie la mémoire par le nombre d'échelles — exactement
l'inverse de l'objectif `d ≤ 8` du goulot Edge. ASFF n'empile que des
**cartes de poids à 1 canal** (beaucoup plus légères), la fusion finale
étant une simple somme pondérée sans jamais concaténer les features
elles-mêmes.

**Paramètre `reference_index`** : l'échelle dont la résolution sert de
référence spatiale (les autres sont upsamplées vers elle). Par défaut
**`-1`** (échelle la plus profonde/petite) plutôt que `0` (la plus fine) :
les convolutions de poids tournent à la résolution de référence, donc
choisir la plus fine ferait tourner ces convs sur la plus grande carte
spatiale pour *chaque* échelle — coût de calcul edge inutilement élevé.

Si le backbone ne supporte pas `extract_multi_scale_features` (cas
`simple_cnn`), `net/pipeline.py` bascule automatiquement en mode
mono-échelle (repli silencieux, sans erreur).

### 2.3 Goulot d'étranglement Edge — `net/edge_bottleneck.py`

Implémente exactement §3.1 : attention **Squeeze-and-Excitation** (pondère
dynamiquement les canaux pertinents, atténue le bruit de fond marin) suivie
d'une **convolution 1×1** qui réduit à `d ≤ 8` canaux.

```python
class SEBlock(nn.Module):
    # AdaptiveAvgPool2d -> FC -> ReLU -> FC -> Sigmoid -> pondération canal

class EdgeBottleneck(nn.Module):
    def forward(self, x):
        x = self.se(x)              # pondération SE
        z = self.proj(x)            # Conv1x1 -> d canaux
        return z
```

### 2.4 Quantification FSQ + STE — `net/fsq.py`

Implémente la formule exacte du §3.1 :

```
z̃ᵢ = round( (Lᵢ-1)/2 · tanh(zᵢ) ),  z̃ᵢ ∈ {-(Lᵢ-1)/2, ..., +(Lᵢ-1)/2}
```

```python
class FSQ(nn.Module):
    def forward(self, z):
        bound = self.bounds.view(1, -1, 1, 1)   # (Lᵢ-1)/2 par dimension
        z_bounded = torch.tanh(z) * bound
        z_rounded = torch.round(z_bounded)
        # Straight-Through Estimator :
        z_q = z_bounded + (z_rounded - z_bounded).detach()
        return z_q
```

Le **STE** (Straight-Through Estimator) est la clé de la
différentiabilité : en forward, on utilise `z_rounded` (la valeur
discrète réelle) ; en backward, le terme `(z_rounded - z_bounded).detach()`
a un gradient nul, donc le gradient qui remonte est celui de `z_bounded`
(continu) — exactement `∂L/∂z̃ = ∂L/∂z` comme prescrit par le document.

Pas de dictionnaire explicite, pas de perte d'engagement (*commitment
loss*) : le dictionnaire implicite est le produit cartésien des niveaux
`∏Lᵢ`, ce qui élimine le *codebook collapse* par construction.

Quatre configurations de codebook sont fournies dans `configs_yaml/` :
16 bits (`L=[8,8,8,5,5,5]`, d=6), 14 bits nominal (`L=[8,8,8,6,5]`, d=5),
10 bits (`L=[8,5,5,5]`, d=4), 5 bits (`L=[2,2,2,2,2]`, d=5).

### 2.5 Mappage binaire — `net/bit_mapping.py`

Convertit chaque valeur FSQ entière en une séquence de bits, et inversement.

**Piège découvert et corrigé** : la borne théorique `(Lᵢ-1)/2` est un
**demi-entier** quand `Lᵢ` est pair (ex. L=8 → borne 3.5), mais FSQ produit
en réalité des valeurs **entières** (`torch.round` arrondit toujours à
l'entier le plus proche, quelle que soit la borne). Utiliser ce décalage
fractionnaire pour calculer l'index binaire provoquait un arrondi
supplémentaire (*banker's rounding* sur les valeurs `.5`) qui perdait 0.5
d'information à chaque fois → le roundtrip encode/decode n'était plus exact.

**Correction** : décalage par l'**entier le plus proche** de la borne
(`round(bound)`), pas la borne fractionnaire elle-même :

```python
self._int_offset = [round((l - 1) / 2.0) for l in levels]
self._range_size = [2 * off + 1 for off in self._int_offset]
self.bits_per_dim = [ceil(log2(r)) for r in self._range_size]
```

Effet de bord : pour les niveaux pairs, ça couvre une plage légèrement plus
large que `Lᵢ` (9 valeurs au lieu de 8 pour L=8), donc **1 bit de plus** par
dimension paire — le budget de bits nominal passe de 14 à 18 bits/token
pour la configuration nominale. Compromis fidélité/bande passante assumé
pour garantir un roundtrip exact.

**Straight-Through Estimator sur `encode()` ET `decode()`** :

- `encode()` : `idx_ste = idx_raw + (idx_rounded - idx_raw).detach()`, puis
  un proxy continu diffusé sur les bits (`idx_proxy`) permet au gradient de
  remonter jusqu'à `zᵢ` malgré l'extraction de bits individuels
  (non-différentiable en soi).
- `decode()` : `bits_ste = bits + (bits_hard - bits).detach()` — **bug
  corrigé** : la version initiale utilisait un seuillage dur
  `(bits > 0.5)` sans aucun STE, ce qui bloquait *totalement* le gradient
  côté réception. Critique car ça aurait empêché la Phase 2B (mise à jour
  conjointe encodeur/décodeur).

**Compatibilité ONNX** : `_int_to_bits` utilisait des opérateurs bitwise
(`>>`, `&`) — testé et confirmé **non exportables** (ni exporteur legacy
TorchScript, ni le nouvel exporteur `dynamo`, échec sur
`aten::__rshift__`/`aten::__and__`). Remplacés par une décomposition
mathématiquement équivalente en division entière + modulo :

```python
bit_k = torch.div(idx, 2 ** k, rounding_mode='floor') % 2
```

---

## 3. Canal physique optique sous-marin

Implémenté dans `net/channel.py::Channel`, modèle IM/DD NRZ-OOK
différentiable de bout en bout (§3.2).

### 3.1 Gain de canal composite `h = h_c · h_t · h_p`

**Atténuation Beer-Lambert** (déterministe) :

```python
h_c = exp(-c_e · L)   # c_e ≈ 0.151 m⁻¹ pour eau claire
```

**Turbulence océanique log-normale** (régime de faible turbulence,
moyenne d'ouverture avec la lentille réceptrice) :

```python
T ~ N(μ_T, σ_T²),  μ_T = -σ_T²/2
h_t = exp(T)
```

**Erreur de pointage géométrique** (distribution de Rayleigh) :

```python
h_p = A₀ · exp(-2r_d² / w_Leq²),  A₀ = erf(ν)²,  ν = D_r·π / (2√2·w_Leq)
```

**Bug physique trouvé et corrigé** : `r_d` (déviation radiale) était tiré
d'une loi normale **standard** (écart-type = 1, sans dimension physique),
alors que `w_Leq` (rayon du faisceau, ex 5 cm) est bien plus petit —
`exp(-2·r_d²/w_Leq²)` s'écrasait quasi toujours à zéro, donnant `h ≈ 0`
systématiquement (BER ≈ 0.5 même à 1 m de distance). Le document §3.2
précise pourtant que `r_d` suit une loi liée à `σ_θt` (**gigue angulaire**,
en radians) — approximation *small-angle* standard : déviation radiale ≈
angle × distance. Paramètre `sigma_theta_t` ajouté (`config.py`, défaut
`2e-3` rad) :

```python
r_d = self.sigma_theta_t * link_distance * |N(0,1)|
```

**Variable discrète `h_a` intentionnellement absente** : dans le modèle
original d'Ijeh et al., `h_a ∈ {0,1}` code une interruption binaire quand
l'angle d'arrivée dépasse le champ de vue de la lentille. Le document
justifie sa suppression (champ de vue suffisant, `Pr(θ_a > θ_FoV) ≈ 0`)
pour garder un canal **entièrement différentiable** — implémenté
conformément, jamais réintroduit.

### 3.2 Bruit SiPM post-détection

```python
σ_n² = R_L² · 2e·G·B_e·(I_s + I_d + I_b) + 4·k_B·T_e·B_e/R_L
```

(grenaille signal-dépendante + bruit thermique TIA)

### 3.3 Seuil optimal et BER exact

```python
γ_th = (r̄₀·σ_n1 + r̄₁·σ_n0) / (σ_n0 + σ_n1)
p = ¼·erfc((γ_th - r̄₀)/√(2σ_n0²)) + ¼·erfc((r̄₁ - γ_th)/√(2σ_n1²))
```

Calculé **par élément** (dépend de `h`, donc varie spatialement dans
l'image/le batch), pas une approximation par seuil empirique — la formule
existait dès le début mais n'était pas branchée dans `forward()` ; c'est
désormais le chemin de calcul réel.

**Bug de calibration trouvé et corrigé** : les valeurs par défaut
`R_L=1e4 Ω, G=1e6` donnaient un bruit de grenaille (`∝ R_L²·G`) démesuré
face au signal, quelle que soit la distance (BER non-monotone, parfois
BER(100m) < BER(50m) — physiquement impossible). Recalibré à `R_L=1e3 Ω, G=1e5`, donnant une courbe BER-vs-distance cohérente (vérifiée par test) :


| Distance | BER théorique                         |
| -------- | -------------------------------------- |
| 1 m      | ~0.0003%                               |
| 20 m     | ~24.5%                                 |
| 60 m     | ~49.4% (saturation, effet falaise §5) |

### 3.4 Compatibilité GPU

`Channel.device` était un simple attribut Python figé à la construction
(`cfg.device`), **pas un buffer `nn.Module`** — un buffer suit
automatiquement un `.to(cuda)` ultérieur, un attribut Python non. Risque
concret : construire le pipeline, charger un checkpoint (CPU), puis
déplacer vers GPU (`pipeline.to('cuda')`) laissait le canal générer son
bruit sur CPU pendant que le reste tournait sur GPU → crash de mismatch de
device. **Corrigé** en dérivant le device dynamiquement de `bits.device`
(le tenseur d'entrée réel) à chaque appel de `forward()`, plutôt que de
l'attribut figé.

---

## 4. Récepteur

### 4.1 Écrêtage strict — `net/latent_clamp.py`

Avant tout traitement, force chaque dimension démappée dans ses bornes FSQ
d'entraînement (§3.3) :

```python
ẑ_clamped,i = clamp(ẑ_i, -(Lᵢ-1)/2, +(Lᵢ-1)/2)
```

Empêche les activations aberrantes causées par des bits inversés de
déstabiliser les couches de normalisation en aval.

### 4.2 IRSNet — `net/irsnet.py`

U-Net résiduel multi-échelle (§3.3), débruite le tenseur latent bruité par
le canal. Architecture : `stem → [ResBlock, Down]×depth → bottleneck → [Up, ResBlock]×depth → head`, connexions résiduelles à chaque niveau.

**Bug corrigé** : pour des résolutions d'entrée non divisibles par
`2^depth` (ex. 111×111), un `Down` (stride 2, arrondi par troncature) suivi
d'un `Up` (stride 2 exact) ne redonnait pas exactement la taille de la
*skip connection* correspondante → crash à l'addition. Corrigé en recalant
`x` sur la taille de `skip` par interpolation bilinéaire si les formes ne
correspondent pas exactement.

### 4.3 Générateur visuel G_ψ — `net/generator.py`

Décode les features restaurées en image reconstruite via des convolutions
transposées (`UpBlock` = `ConvTranspose2d + BatchNorm + ReLU`), sortie
`Sigmoid` pour rester dans `[0,1]³ˣᴴˣᵂ` comme prescrit.

**Bug corrigé** : le nombre de blocs d'upsampling était figé à la
construction, mais différents backbones sous-échantillonnent différemment
(simple_cnn ×8, YOLO26/SegFormer ×32 à l'échelle profonde) — mismatch de
résolution en sortie. Corrigé en deux temps :

1. `net/pipeline.py::_infer_downsample_factor` calcule le facteur réel par
   une passe factice à la construction, et dimensionne
   `num_upsamples = log2(downsample_factor)` en conséquence.
2. `Generator.forward(z, target_size=...)` accepte un resize bilinéaire
   final de sécurité si un écart résiduel subsiste (résolutions non
   exactement alignées sur une puissance de 2).

### 4.4 Têtes de tâches aval — `net/heads.py`

**Insight architectural clé** : ces têtes traitent `x̂` (l'image RGB
**reconstruite**), pas les features de l'émetteur — elles sont donc
**totalement indépendantes** du backbone choisi côté émetteur. Un seul jeu
de têtes suffit quel que soit le backbone émetteur (YOLO26 ou SegFormer).

Deux implémentations disponibles, sélectionnées via `config.use_real_task_heads` :


|                | `SimpleSegmentationHead`/`SimpleClassificationHead` | `SegFormerSegmentationHead`/`SegFormerClassificationHead` |
| -------------- | --------------------------------------------------- | --------------------------------------------------------- |
| Poids          | CNN léger, non pré-entraîné                     | Encodeur MiT-B5 complet + tête officielle`transformers`  |
| Usage          | Dev rapide, tests                                   | §3.4 du document — têtes "réelles"                    |
| Coût mémoire | Faible                                              | ~80M params ×2 (seg+cls)                                 |

**Pourquoi pas la vraie tête `Segment26` de YOLO26** : elle fait de la
**segmentation d'instance** (boîtes + masques par objet + NMS), pas de la
segmentation **sémantique dense** pixel-par-pixel comme notre tâche
CoralMask — la réutiliser demanderait un adaptateur non trivial
(post-traitement NMS, fusion de masques d'instances en carte sémantique)
hors du cadre d'une simple tête de sortie. D'où le choix de SegFormer
(`SegformerForSemanticSegmentation`/`SegformerForImageClassification`),
architecturalement adapté aux deux tâches dès le départ, testé
numériquement (sortie native `H/4,W/4`, upsamplée vers la résolution de `x̂`).

---

## 5. Orchestration du pipeline complet

`net/pipeline.py::UWOCSemComPipeline` assemble tous les modules. Interface
principale :

```python
pipeline = UWOCSemComPipeline(cfg)
out = pipeline(x, link_distance=10.0, phase='phase2a')
# out = {'x_hat', 'seg_logits', 'cls_logits', 'z_q', 'z_restored', 'ber'}
```

Le paramètre `phase` contrôle le comportement du canal et de l'attaque de
masque (voir §6) :


| `phase`     | Canal actif ?                    | MATK actif ? |
| ----------- | -------------------------------- | ------------ |
| `'phase1'`  | Non (régime analogique parfait) | Non          |
| `'phase2a'` | Oui                              | Oui          |
| `'phase3'`  | Oui                              | Non          |
| `'phase4'`  | Oui                              | Non          |

```python
def forward(self, x, link_distance, phase='phase1'):
    z_q = self.encode(x)                        # backbone→ASFF→bottleneck→FSQ
    if phase == 'phase1':
        z_hat = z_q; p = 0
    else:
        bits = self.bit_mapper.encode(z_q)
        b_hat, p = self.channel(bits, link_distance)
        z_hat = self.bit_mapper.decode(b_hat)
    z_clamped = clamp_latent(z_hat, self.fsq.levels)
    if phase == 'phase2a':
        z_clamped = self.matk(z_clamped, p)      # attaque de masque
    z_restored = self.irsnet(z_clamped)
    x_hat = self.generator(z_restored, target_size=x.shape[-2:])
    return {'x_hat': x_hat,
            'seg_logits': self.seg_head(x_hat),
            'cls_logits': self.cls_head(x_hat),
            'z_q': z_q, 'z_restored': z_restored, 'ber': p}
```

---

## 6. Entraînement en 4 phases (AMP)

Implémenté dans `engine/`, orchestré par `engine/amp_scheduler.py`. Suit
exactement le protocole §4.

### Phase 1 — Extraction de Caractéristiques (FE) — `engine/phase1_fe.py`

Entraînement conjoint encodeur+décodeur, **sans bruit de canal**
(`phase='phase1'`). Têtes de tâches aval gelées. Perte : `L_P1 = ‖x-x̂‖₁`.

### Phase 2 — Renforcement de Robustesse (RE) — `engine/phase2_re.py`

**Étape 2A** : encodeur **gelé** (backbone + **fusion ASFF** + edge
bottleneck), IRSNet+Generator entraînés sous bruit binaire (BFP) et attaque
de masque (MATK, ratio `2p`). Perte :
`L_P2A = ‖x-x̂‖₁ + β·‖ẑ_restored - z_q‖₁`.

**Bug corrigé** : la liste des modules à geler omettait `multiscale_fusion`
— ASFF restait entraînable par erreur pendant cette phase censée geler
tout l'encodeur.

### Phase 3 — Alignement Entraînement-Test (TTA) — `engine/phase3_tta.py`

Même perte que 2A, mais **MATK désactivé** (`phase='phase3'`) — supprime le
décalage de distribution introduit par le masque artificiel.

### Phase 4 — Adaptation des Têtes de Tâches — `engine/phase4_task.py`

Gel **complet** de la chaîne codec+canal (backbone, ASFF, bottleneck, FSQ,
IRSNet, Generator). Seules les têtes `seg_head`/`cls_head` sont
entraînées. Perte (`losses/latent_align.py::compute_phase4_loss`) :

```
L_P4 = CE(seg) + λ_dice·Dice(seg) + λ_cls·CE(cls)
```

**Fonctionnalité ajoutée** : la perte de classification (`cls_logits`,
`cls_labels`) était absente de la version initiale de `compute_phase4_loss`
(seule la segmentation était optimisée) — ajoutée, paramètre `lambda_cls`
optionnel pour compatibilité arrière.

### Format des batches

`datasets/coral_dataset.py::CoralDataset`/`DummyCoralDataset` renvoient un
dictionnaire `{'image', 'mask', 'label', 'filename'}` par échantillon —
tous les fichiers `engine/phase*.py` unpacktent via `batch['image']`,
`batch['mask']`, `batch['label']` de façon cohérente.

### `DummyCoralDataset`

En l'absence du vrai dataset (branchement volontairement différé),
`datasets/coral_dataset.py::DummyCoralDataset` génère des images/masques/
labels aléatoires de la bonne forme — permet de vérifier que **toute la
mécanique tourne** (les 4 phases, `main.py` intégralement) sans donner de
résultats scientifiquement valides.

---

## 7. Export ONNX et déploiement edge

`net/emitter.py::Emitter` isole la chaîne **émetteur seul**
(`backbone → ASFF → edge_bottleneck → FSQ → bit_mapper.encode`), exportable
en ONNX pour déploiement sur la sonde.

**Pourquoi un module séparé plutôt que le pipeline entier** :

- Le canal physique contient du bruit aléatoire (`torch.randn`/`torch.normal`),
  non reproductible une fois figé dans un graphe ONNX déterministe.
- Le récepteur (IRSNet, Generator, têtes) tourne côté **station de
  surface**, pas sur la sonde — inutile de l'embarquer.

```python
emitter = Emitter.from_pipeline(pipeline)   # partage les poids, ne les copie pas
torch.onnx.export(emitter, x, 'emitter.onnx', ...)
```

Script CLI : `scripts/export_emitter_onnx.py`.

**Validation numérique** : sur `simple_cnn`, l'export est **bit-exact**
(diff PyTorch vs ONNX Runtime = 0.0). Sur YOLO26+ASFF (backbone profond),
quelques bits peuvent différer (généralement <1%, jusqu'à ~11% observé
selon l'initialisation des poids) — **pas un bug d'export**, mais un
phénomène documenté : PyTorch et ONNX Runtime ne diffèrent que de `~1e-6`
sur les features (bruit numérique cross-runtime normal), mais cette
différence infime peut faire basculer une valeur FSQ très proche d'une
frontière d'arrondi (`x.5`), provoquant une cascade de quelques bits
différents dans l'encodage binaire de cette dimension. C'est littéralement
le phénomène de sensibilité aux sauts de bits (MSB) que le document §1
identifie comme verrou majeur — ici la source du "bruit" est le
non-déterminisme numérique inter-runtime plutôt que le canal physique.

---

## 8. Compression du modèle

`utils/model_compression.py` — profilage et compression pour cible edge.

### Profilage (`profile_model`, `count_flops`, `count_parameters`)

Comptage manuel des FLOPs/MACs via hooks forward sur `Conv2d`/`Linear`,
sans dépendance externe. `count_parameters` mesure la taille **réellement
sérialisée** (`state_dict`), fiable même pour un modèle quantifié (les
poids packés ne sont pas de simples `nn.Parameter`, `numel()` seul sous-estime).

### Quantization statique calibrée (`quantize_backbone_static`)

Flux PyTorch eager-mode complet (`QuantStub`/`DeQuantStub`, `prepare`,
calibration, `convert`), **fonctionne et testé** sur nos propres modules
(`Generator` : ×3 de compression, forward correct).

**Limite constatée sur YOLO26** (documentée, pas cachée) :

1. `SiLU` n'a pas de noyau quantized CPU natif → contourné avec un wrapper
   dequant→SiLU→quant (`_DequantActQuant`, technique standard).
2. Les **additions résiduelles internes d'ultralytics**
   (`x + self.cv2(self.cv1(x))`, dans *leur* code) n'ont pas de noyau
   quantized pour l'addition simple entre tenseurs quantifiés — pas
   contournable de l'extérieur sans monkey-patcher leur bibliothèque.

**Conclusion pratique** : un vrai déploiement edge de YOLO26 quantifié
nécessiterait un export ONNX + quantization via **ONNX Runtime** (testé
séparément, fonctionne réellement sur YOLO26 : ×3.3 de compression,
dynamique et statique, contrairement à PyTorch natif) — mais sur CPU x86 de
ce container de dev, la quantization ONNX Runtime a paradoxalement **ralenti**
l'inférence plutôt que de l'accélérer (absence d'optimisation pour ce
device/cette taille de modèle) — un rappel qu'il faut toujours benchmarker
sur le matériel cible réel, pas supposer.

### Pruning non structuré (`prune_backbone_unstructured`)

Met à zéro une fraction des poids (`torch.nn.utils.prune`, norme L1).
Réduit l'empreinte mémoire de stockage, **pas** les FLOPs/latence sans
backend creux dédié.

## 10. Structure du dépôt

```
uwoc-semcom/
├── main.py                        # point d'entrée entraînement (AMP)
├── config.py                      # configuration centrale
├── net/
│   ├── backbone.py                # YOLO26 / SegFormer MiT-B5 / simple_cnn
│   ├── multiscale_fusion.py       # ASFF
│   ├── edge_bottleneck.py         # SE + Conv1x1
│   ├── fsq.py                     # quantification FSQ + STE
│   ├── bit_mapping.py             # encode/decode binaire
│   ├── channel.py                 # canal physique IM/DD UWOC
│   ├── latent_clamp.py            # écrêtage strict récepteur
│   ├── irsnet.py                  # U-Net résiduel débruiteur
│   ├── generator.py               # décodeur visuel G_ψ
│   ├── heads.py                   # têtes segmentation/classification
│   ├── mask_attack.py             # MATK (Phase 2A)
│   ├── pipeline.py                # UWOCSemComPipeline (assemblage complet)
│   └── emitter.py                 # émetteur seul, exportable ONNX
├── datasets/coral_dataset.py      # CoralDataset (réel) + DummyCoralDataset
├── engine/
│   ├── amp_scheduler.py           # sélection de phase par époque
│   └── phase{1,2,3,4}_*.py        # boucles d'entraînement par phase
├── losses/
│   ├── reconstruction.py          # L1, PSNR, SSIM
│   ├── latent_align.py            # pertes Phase 1/2A/3/4
│   └── task_losses.py             # CE/Dice/CE (utilitaires)
├── evaluation/
│   ├── ber_curves.py              # tracé BER vs distance
│   └── rate_distortion_fit.py     # fit du modèle ξ(ρ,γ) (§5)
├── utils/
│   ├── checkpoint.py, freeze_utils.py, logger.py, metrics.py
│   └── model_compression.py       # profilage, quantization, pruning
├── scripts/export_emitter_onnx.py # CLI export ONNX émetteur
├── configs_yaml/                  # 4 configs codebook + config canal
└── tests/                         # 9 fichiers, 25 tests unitaires
```

---

## 11. Comment lancer le projet

```bash
pip install -r requirements.txt

# Entraînement complet (4 phases, config par défaut)
python main.py

# Export ONNX de l'émetteur seul
python scripts/export_emitter_onnx.py --output emitter.onnx --image_size 128

# Suite de tests complète
export PYTHONPATH=.
for f in tests/test_*.py; do python3 "$f"; done
```

`config.py` centralise tous les hyperparamètres (backbone, niveaux FSQ,
paramètres du canal physique, têtes de tâches). Voir les commentaires
inline pour chaque valeur — notamment les notes de calibration sur les
paramètres du canal (§3 de ce document).

---
