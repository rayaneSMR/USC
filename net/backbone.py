"""
Extracteur de caractéristiques de l'émetteur (§2 du pipeline UWOC-SemCom) :
Backbone YOLO26 (segmentation) ou SegFormer MiT-B5.

Implémentation réelle, alignée sur le notebook de référence
(UWOC_YOLO26_SegFormer_OptionA_NoHa.ipynb, section "4. Model wrappers"),
adaptée à notre pipeline (edge bottleneck + FSQ à dimension unique d ≤ 8,
au lieu du canal multi-échelle par liste utilisé dans le notebook).

Choix d'implémentation :
- YOLO26SplitSeg : `ultralytics.YOLO(weights).model`, un forward pre-hook
  capture l'entrée de la tête `Segment26` (dernier module de `.model.model`).
  Cette entrée est en réalité une LISTE de 3 cartes multi-échelle (style
  FPN P3/P4/P5), vérifié empiriquement :
      feat[0]: (B,  64, H/8,  W/8)
      feat[1]: (B, 128, H/16, W/16)
      feat[2]: (B, 256, H/32, W/32)
  Par défaut on ne garde QUE l'échelle la plus profonde (feat[-1], la plus
  riche sémantiquement, la plus petite spatialement) pour alimenter le
  goulot d'étranglement Edge à sortie unique (§3.1 du pipeline). Les 3
  échelles restent accessibles via `extract_multi_scale_features()` pour
  une extension future vers un bottleneck multi-échelle.
- SegFormerMiTB5 : `transformers.SegformerModel` (encodeur seul, sans tête
  de décodage — celle-ci est remplacée dans notre pipeline par IRSNet +
  Generator côté récepteur). `output_hidden_states=True` renvoie déjà les
  4 échelles en format spatial (B,C,H,W) (vérifié : pas de tokens aplatis
  à extraire). Par défaut on garde hidden_states[-1] (le plus profond,
  512 canaux pour MiT-B5).
"""
import torch
import torch.nn as nn


class SimpleCNNBackbone(nn.Module):
    """Backbone CNN léger de repli, utile pour les tests rapides sans
    télécharger de poids externes. Produit (B, C, H/8, W/8)."""

    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class YOLO26Backbone(nn.Module):
    """
    Backbone+neck de YOLO26-seg (émetteur), tête `Segment26` détachée.

    weights: nom/chemin des poids ultralytics, ex 'yolo26n-seg.pt' (nano)
             ou 'yolo26s-seg.pt' (small). Cherchés d'abord localement
             (répertoire courant, puis `local_weights_dir`) avant tout essai
             de téléchargement depuis les releases GitHub d'Ultralytics --
             utile pour un déploiement offline sur la sonde/edge, où l'accès
             réseau n'est pas garanti.
    local_weights_dir: répertoire local à vérifier en priorité pour les
             poids (ex '/opt/uwoc/weights'). Si None, seul le répertoire
             courant est vérifié avant le téléchargement.
    feature_scale: index de l'échelle à retourner par `forward()`
             (-1 = la plus profonde/petite spatialement, comportement
             par défaut de notre pipeline à bottleneck simple).
    """

    def __init__(self, weights='yolo26n-seg.pt', feature_scale=-1, freeze=False,
                 local_weights_dir=None):
        super().__init__()
        from ultralytics import YOLO  # import différé : dépendance lourde optionnelle

        resolved_weights = self._resolve_weights_path(weights, local_weights_dir)
        try:
            self.yolo_model = YOLO(resolved_weights).model
        except Exception as e:
            raise RuntimeError(
                f"Impossible de charger les poids YOLO26 '{weights}' "
                f"(cherché localement puis en ligne). "
                f"Pour un déploiement offline sur edge, placer le fichier "
                f"'{weights}' dans le répertoire courant ou dans "
                f"`local_weights_dir` avant l'exécution. Erreur d'origine: {e}"
            ) from e

        self.head_module = self.yolo_model.model[-1]
        self.feature_scale = feature_scale

        self._captured_inputs = None
        self._hook_handle = self.head_module.register_forward_pre_hook(self._capture_head_input)

        # détermine out_channels par une passe factice
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 384, 384)
            feats = self._forward_capture(dummy)
        self.out_channels = feats[self.feature_scale].shape[1]

        if freeze:
            for p in self.yolo_model.parameters():
                p.requires_grad = False

    @staticmethod
    def _resolve_weights_path(weights, local_weights_dir):
        """Cherche `weights` localement avant de laisser ultralytics tenter
        un téléchargement (qui échouera sans réseau -- voir point 8)."""
        import os
        candidates = [weights]
        if local_weights_dir is not None:
            candidates.insert(0, os.path.join(local_weights_dir, weights))
        for path in candidates:
            if os.path.isfile(path):
                return path
        return weights  # laisse ultralytics gérer le téléchargement

    def _capture_head_input(self, module, inputs):
        self._captured_inputs = inputs[0]
        return inputs

    def _forward_capture(self, x):
        self._captured_inputs = None
        _ = self.yolo_model(x)
        feats = self._captured_inputs
        if feats is None:
            raise RuntimeError(
                "L'entrée de la tête YOLO26 n'a pas été capturée. "
                "Vérifier la version d'ultralytics installée (API interne susceptible de changer)."
            )
        return feats  # liste de tenseurs multi-échelle

    def extract_multi_scale_features(self, x):
        """Retourne la liste complète des cartes multi-échelle (P3, P4, P5)."""
        return self._forward_capture(x)

    def forward(self, x):
        """Retourne une carte de features unique (échelle `feature_scale`)."""
        feats = self._forward_capture(x)
        return feats[self.feature_scale]


class SegFormerMiTB5Backbone(nn.Module):
    """
    Encodeur SegFormer MiT-B5 (émetteur), sans tête de décodage.

    pretrained_name: identifiant HuggingFace (ex 'nvidia/mit-b5') si accès
             réseau disponible, sinon `None` pour instancier une config
             MiT-B5 non pré-entraînée (utile en développement/hors-ligne).
    feature_scale: index de l'échelle retournée par `forward()` (-1 = la
             plus profonde, 512 canaux pour MiT-B5).
    """

    # Hyperparamètres officiels MiT-B5 (transformers.SegformerConfig)
    MIT_B5_CONFIG = dict(
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

    def __init__(self, pretrained_name=None, feature_scale=-1, freeze=False):
        super().__init__()
        from transformers import SegformerModel, SegformerConfig

        if pretrained_name is not None:
            self.encoder = SegformerModel.from_pretrained(pretrained_name)
        else:
            config = SegformerConfig(**self.MIT_B5_CONFIG)
            self.encoder = SegformerModel(config)

        self.feature_scale = feature_scale
        self.out_channels = self.MIT_B5_CONFIG['hidden_sizes'][feature_scale]

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def extract_multi_scale_features(self, x):
        """Retourne la liste complète des 4 échelles (B,C,H,W) déjà spatiales."""
        out = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        return list(out.hidden_states)

    def forward(self, x):
        feats = self.extract_multi_scale_features(x)
        return feats[self.feature_scale]


class Backbone(nn.Module):
    """
    Wrapper générique unifié. backbone_name ∈
        {'yolo26', 'segformer_mitb5', 'simple_cnn'}
    """

    def __init__(self, backbone_name='simple_cnn', pretrained=True, freeze=False,
                 yolo_weights='yolo26n-seg.pt', segformer_pretrained_name=None,
                 feature_scale=-1, local_weights_dir=None):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == 'simple_cnn':
            self.model = SimpleCNNBackbone()
            self.out_channels = 256
            if freeze:
                for p in self.model.parameters():
                    p.requires_grad = False

        elif backbone_name == 'yolo26':
            self.model = YOLO26Backbone(
                weights=yolo_weights, feature_scale=feature_scale, freeze=freeze,
                local_weights_dir=local_weights_dir,
            )
            self.out_channels = self.model.out_channels

        elif backbone_name == 'segformer_mitb5':
            self.model = SegFormerMiTB5Backbone(
                pretrained_name=segformer_pretrained_name if pretrained else None,
                feature_scale=feature_scale, freeze=freeze,
            )
            self.out_channels = self.model.out_channels

        else:
            raise ValueError(f"backbone_name inconnu: {backbone_name}")

    def extract_multi_scale_features(self, x):
        if hasattr(self.model, 'extract_multi_scale_features'):
            return self.model.extract_multi_scale_features(x)
        raise NotImplementedError(
            f"extract_multi_scale_features non disponible pour '{self.backbone_name}'"
        )

    def forward(self, x):
        return self.model(x)
