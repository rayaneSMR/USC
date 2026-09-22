"""
Emitter — module émetteur autonome (§2, jusqu'au flux binaire b), conçu
pour être exporté en ONNX en vue d'un déploiement edge réel.

Chaîne couverte :
    x -> Backbone (multi-échelle) -> ASFF -> EdgeBottleneck -> FSQ -> BitMapper.encode -> b

Pourquoi un module séparé plutôt que d'exporter le pipeline entier :
- Le canal physique (net/channel.py) contient du bruit aléatoire, non
  reproductible/déterministe une fois figé dans un graphe ONNX.
- Le récepteur (IRSNet, Generator, têtes) tourne côté station de surface,
  pas sur la sonde -- inutile de l'embarquer dans le binaire edge.
"""
import torch
import torch.nn as nn


class Emitter(nn.Module):
    def __init__(self, backbone, multiscale_fusion, edge_bottleneck, fsq, bit_mapper):
        super().__init__()
        self.backbone = backbone
        self.multiscale_fusion = multiscale_fusion
        self.edge_bottleneck = edge_bottleneck
        self.fsq = fsq
        self.bit_mapper = bit_mapper

    @classmethod
    def from_pipeline(cls, pipeline):
        """Construit un Emitter à partir des sous-modules déjà entraînés
        d'un UWOCSemComPipeline existant (partage les poids, ne les copie pas)."""
        return cls(
            backbone=pipeline.backbone,
            multiscale_fusion=pipeline.multiscale_fusion,
            edge_bottleneck=pipeline.edge_bottleneck,
            fsq=pipeline.fsq,
            bit_mapper=pipeline.bit_mapper,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiscale_fusion is not None:
            multi_feats = self.backbone.extract_multi_scale_features(x)
            feat = self.multiscale_fusion(multi_feats)
        else:
            feat = self.backbone(x)
        z = self.edge_bottleneck(feat)
        z_q = self.fsq(z)
        b = self.bit_mapper.encode(z_q)
        return b
