"""
Orchestrateur global du pipeline sémantique UWOC-SemCom.
Relie l'encodeur (backbone + fusion multi-échelle ASFF + bottleneck edge +
FSQ + bit mapping), le canal optique, et le décodeur (IRSNet + Generator +
têtes de tâches aval).
"""
import torch
import torch.nn as nn

from .backbone import Backbone
from .multiscale_fusion import ASFF
from .edge_bottleneck import EdgeBottleneck
from .fsq import FSQ
from .bit_mapping import BitMapper
from .channel import Channel
from .latent_clamp import clamp_latent
from .irsnet import IRSNet
from .generator import Generator
from .heads import (
    SegFormerSegmentationHead, SegFormerClassificationHead,
    SimpleSegmentationHead, SimpleClassificationHead,
)
from .mask_attack import MATK


class UWOCSemComPipeline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- Émetteur ---
        self.backbone = Backbone(
            backbone_name=cfg.backbone_name,
            pretrained=cfg.backbone_pretrained,
            freeze=cfg.backbone_freeze,
            yolo_weights=cfg.yolo_weights,
            segformer_pretrained_name=cfg.segformer_pretrained_name,
            local_weights_dir=cfg.local_weights_dir,
            feature_scale=-1,
        )

        # Fusion multi-échelle ASFF : certains backbones (simple_cnn) ne
        # supportent pas extract_multi_scale_features -- fallback mono-échelle
        # automatique dans ce cas, sans planter.
        self.use_multiscale_fusion = cfg.use_multiscale_fusion and hasattr(
            self.backbone.model, 'extract_multi_scale_features')
        if self.use_multiscale_fusion:
            in_channels_list = self._infer_multiscale_channels(cfg.image_size)
            self.multiscale_fusion = ASFF(
                in_channels_list, fusion_channels=cfg.fusion_channels,
                reference_index=cfg.fusion_reference_index,
            )
            edge_in_channels = cfg.fusion_channels
        else:
            self.multiscale_fusion = None
            edge_in_channels = self.backbone.out_channels

        self.edge_bottleneck = EdgeBottleneck(in_channels=edge_in_channels, bottleneck_dim=cfg.d)
        self.fsq = FSQ(levels=cfg.levels)
        self.bit_mapper = BitMapper(levels=cfg.levels)

        # --- Canal Physique ---
        self.channel = Channel(cfg)

        # --- Récepteur ---
        self.irsnet = IRSNet(in_channels=cfg.d)

        downsample_factor = self._infer_downsample_factor(cfg.image_size)
        num_upsamples = max(1, round(torch.log2(torch.tensor(float(downsample_factor))).item()))
        self.generator = Generator(in_channels=cfg.d, num_upsamples=num_upsamples)

        if cfg.use_real_task_heads:
            self.seg_head = SegFormerSegmentationHead(
                num_classes=cfg.num_classes_seg,
                pretrained_name=cfg.segformer_head_pretrained_name,
                freeze_encoder=cfg.freeze_task_head_encoders,
            )
            self.cls_head = SegFormerClassificationHead(
                num_classes=cfg.num_classes_cls,
                pretrained_name=cfg.segformer_head_pretrained_name,
                freeze_encoder=cfg.freeze_task_head_encoders,
            )
        else:
            self.seg_head = SimpleSegmentationHead(in_channels=3, num_classes=cfg.num_classes_seg)
            self.cls_head = SimpleClassificationHead(in_channels=3, num_classes=cfg.num_classes_cls)

        # --- Perturbation d'entraînement (Phase 2A) ---
        self.matk = MATK(ratio_factor=2.0)

    # ---- utilitaires de calibration à la construction ----

    def _infer_multiscale_channels(self, image_size):
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            feats = self.backbone.extract_multi_scale_features(dummy)
        self.backbone.train(was_training)
        return [f.shape[1] for f in feats]

    def _infer_downsample_factor(self, image_size):
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            if self.use_multiscale_fusion:
                feats = self.backbone.extract_multi_scale_features(dummy)
                fused = self.multiscale_fusion(feats)
                ref_h = fused.shape[-1]
            else:
                ref_h = self.backbone(dummy).shape[-1]
        self.backbone.train(was_training)
        return max(1, image_size // ref_h)

    # ---- forward ----

    def encode(self, x):
        if self.use_multiscale_fusion:
            multi_feats = self.backbone.extract_multi_scale_features(x)
            feat = self.multiscale_fusion(multi_feats)
        else:
            feat = self.backbone(x)
        z = self.edge_bottleneck(feat)
        z_q = self.fsq(z)
        return z_q

    def forward(self, x, link_distance, phase='phase1', run_heads=True):
        # 1. Extraction et Goulot d'Étranglement (Edge)
        z_q = self.encode(x)

        # Validation stricte de la phase
        allowed_phases = ('phase1', 'phase2a', 'phase2b', 'phase3', 'phase4')
        if phase not in allowed_phases:
            raise ValueError(f"Phase inconnue: '{phase}'. Valeurs permises: {allowed_phases}")

        # 2. Canal Physique
        if phase == 'phase1':
            # Phase 1 : Extraction de caractéristiques (sans bruit)
            z_hat = z_q
            p = torch.zeros(1, device=x.device)
        else:
            # Phases 2, 3, 4 : Modulation IM/DD et Bruit
            bits = self.bit_mapper.encode(z_q)
            b_hat, p = self.channel(bits, link_distance)
            z_hat = self.bit_mapper.decode(b_hat)

        # 3. Récepteur - Écrêtage strict
        z_clamped = clamp_latent(z_hat, self.fsq.levels)

        # Application de l'attaque de masque uniquement en phase 2A
        if phase == 'phase2a':
            self.matk.set_enabled(True)
            z_clamped = self.matk(z_clamped, p)
        else:
            self.matk.set_enabled(False)

        # Restauration Latente
        z_restored = self.irsnet(z_clamped)

        # 4. Génération visuelle et Tâches aval
        x_hat = self.generator(z_restored, target_size=x.shape[-2:])
        if run_heads:
            seg_logits = self.seg_head(x_hat)
            cls_logits = self.cls_head(x_hat)
        else:
            seg_logits = None
            cls_logits = None

        return {
            'x_hat': x_hat,
            'seg_logits': seg_logits,
            'cls_logits': cls_logits,
            'z_q': z_q,
            'z_restored': z_restored,
            'ber': p,
        }
