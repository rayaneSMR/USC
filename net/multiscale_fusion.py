"""
ASFF — Adaptively Spatial Feature Fusion (Liu, Huang, Wang, 2019,
"Learning Spatial Fusion for Single-Shot Object Detection").

S'insère entre le backbone (YOLO26 / SegFormer, sorties multi-échelle) et
le Goulot d'Étranglement Edge (§3.1 du pipeline) :

    Backbone --[N échelles]--> ASFF --[1 tenseur fusionné]--> EdgeBottleneck -> FSQ

Contrainte edge respectée : contrairement à BiFPN/PANet/cross-attention
classique, les features complètes de chaque échelle ne sont JAMAIS
concaténées (ce qui multiplierait la mémoire par N). Seules de petites
cartes de poids à 1 canal par échelle sont empilées puis passées dans un
softmax le long de la dimension "échelle" — la fusion finale est une
simple somme pondérée spatiale, pas un concat de features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFF(nn.Module):
    """
    in_channels_list: liste des nb de canaux de chaque échelle du backbone,
        ex [64, 128, 256] pour YOLO26 ou [64, 128, 320, 512] pour SegFormer.
    fusion_channels: nb de canaux communs après projection 1x1 (C_ref).
    reference_index: index de l'échelle dont la résolution spatiale sert de
        référence pour l'upsampling des autres échelles. Par défaut -1
        (échelle la plus profonde/petite) pour respecter la contrainte edge :
        les convs `weight_conv` (3x3) tournent à la résolution de référence,
        donc choisir l'échelle la plus fine comme référence ferait tourner
        ces convs sur la plus grande carte spatiale pour CHAQUE échelle
        (y compris les profondes upsamplées) -- coût inutilement élevé pour
        un émetteur edge. Mettre 0 (échelle la plus fine) uniquement si le
        détail spatial fin prime sur le coût de calcul.
    """

    def __init__(self, in_channels_list, fusion_channels=64, reference_index=-1):
        super().__init__()
        self.n_scales = len(in_channels_list)
        self.fusion_channels = fusion_channels
        self.reference_index = reference_index

        # Projection 1x1 par échelle vers un nombre de canaux commun (léger)
        self.proj = nn.ModuleList([
            nn.Conv2d(c_in, fusion_channels, kernel_size=1)
            for c_in in in_channels_list
        ])

        # Branche de poids : une petite conv 3x3 par échelle, sortie 1 canal.
        # Coût négligeable (pas de features complètes impliquées ici).
        self.weight_conv = nn.ModuleList([
            nn.Conv2d(fusion_channels, 1, kernel_size=3, padding=1)
            for _ in range(self.n_scales)
        ])

    def forward(self, features: list) -> torch.Tensor:
        """
        features: liste de N tenseurs (B, C_i, H_i, W_i), une échelle chacun
                  (sortie de `Backbone.extract_multi_scale_features`).
        Retourne un tenseur fusionné unique (B, fusion_channels, H_ref, W_ref).
        """
        assert len(features) == self.n_scales, (
            f"ASFF configuré pour {self.n_scales} échelles, "
            f"reçu {len(features)}."
        )
        ref_size = features[self.reference_index].shape[-2:]

        projected = []
        for i, feat in enumerate(features):
            p = self.proj[i](feat)
            if p.shape[-2:] != ref_size:
                p = F.interpolate(p, size=ref_size, mode='bilinear', align_corners=False)
            projected.append(p)

        # Cartes de poids brutes (B, 1, H_ref, W_ref) par échelle
        raw_weights = [self.weight_conv[i](projected[i]) for i in range(self.n_scales)]
        stacked_weights = torch.cat(raw_weights, dim=1)          # (B, N, H_ref, W_ref)
        norm_weights = F.softmax(stacked_weights, dim=1)         # softmax le long des échelles

        # Somme pondérée spatiale : pas de concat des features elles-mêmes
        fusion = torch.zeros_like(projected[self.reference_index])
        for i in range(self.n_scales):
            w_i = norm_weights[:, i:i + 1, :, :]                  # (B, 1, H_ref, W_ref)
            fusion = fusion + w_i * projected[i]

        return fusion
