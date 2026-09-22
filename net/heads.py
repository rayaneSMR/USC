"""
Têtes de tâches aval (§3.4) : segmentation sémantique (CoralMask) et
classification, appliquées sur l'image reconstruite x_hat.

INSIGHT ARCHITECTURAL : ces têtes traitent x_hat (l'image RGB reconstruite),
PAS les features transmises par l'émetteur -- elles sont donc totalement
INDÉPENDANTES du backbone choisi côté émetteur (YOLO26 ou SegFormer). Un
seul jeu de têtes (basées sur SegFormer, qui dispose de têtes officielles
de segmentation ET de classification) suffit quel que soit le backbone
émetteur.

Pourquoi pas la tête `Segment26` de YOLO26 comme "vraie" tête : elle fait de
la segmentation D'INSTANCE (boîtes + masques par objet + NMS), pas de la
segmentation SÉMANTIQUE dense pixel par pixel comme notre tâche CoralMask --
la réutiliser nécessiterait un adaptateur non trivial hors du cadre d'une
simple tête de sortie.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSegmentationHead(nn.Module):
    """Tête de segmentation CNN légère (repli rapide/dev, pas de dépendance
    lourde). SegFormerSegmentationHead est la tête "réelle" recommandée."""

    def __init__(self, in_channels=3, num_classes=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, x_hat):
        return self.net(x_hat)


class SimpleClassificationHead(nn.Module):
    """Tête de classification CNN légère (repli rapide/dev)."""

    def __init__(self, in_channels=3, num_classes=10, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x_hat):
        feat = self.backbone(x_hat).flatten(1)
        return self.fc(feat)


# Hyperparamètres officiels MiT-B5 (mêmes que net/backbone.py:SegFormerMiTB5Backbone)
_MIT_B5_CONFIG = dict(
    num_channels=3,
    num_encoder_blocks=4,
    depths=[3, 6, 40, 3],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[64, 128, 320, 512],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    num_attention_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
)


class SegFormerSegmentationHead(nn.Module):
    """
    Vraie tête de segmentation sémantique SegFormer (§3.4), appliquée
    directement sur x_hat -- encodeur MiT-B5 propre à cette tête (indépendant
    de celui de l'émetteur) + SegformerDecodeHead officiel. Upsample la
    sortie (nativement H/4, W/4) vers la résolution de x_hat.
    """

    def __init__(self, num_classes=2, pretrained_name=None, freeze_encoder=False):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained_name is not None:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            config = SegformerConfig(**_MIT_B5_CONFIG, num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(config)

        if freeze_encoder:
            for p in self.model.segformer.parameters():
                p.requires_grad = False

    def forward(self, x_hat):
        out = self.model(pixel_values=x_hat)
        logits = out.logits
        logits = F.interpolate(logits, size=x_hat.shape[-2:], mode='bilinear', align_corners=False)
        return logits


class SegFormerClassificationHead(nn.Module):
    """
    Vraie tête de classification SegFormer (§3.4), appliquée directement sur
    x_hat -- encodeur MiT-B5 propre + tête linéaire officielle
    (`SegformerForImageClassification`).
    """

    def __init__(self, num_classes=10, pretrained_name=None, freeze_encoder=False):
        super().__init__()
        from transformers import SegformerConfig, SegformerForImageClassification

        if pretrained_name is not None:
            self.model = SegformerForImageClassification.from_pretrained(
                pretrained_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            config = SegformerConfig(**_MIT_B5_CONFIG, num_labels=num_classes)
            self.model = SegformerForImageClassification(config)

        if freeze_encoder:
            for p in self.model.segformer.parameters():
                p.requires_grad = False

    def forward(self, x_hat):
        out = self.model(pixel_values=x_hat)
        return out.logits


# Alias rétro-compatibles (ancien nom utilisé par pipeline.py/tests)
SegmentationHead = SegFormerSegmentationHead
ClassificationHead = SegFormerClassificationHead
