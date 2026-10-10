"""
Générateur de décodage visuel G_psi (§3.4). Décode les caractéristiques
latentes restaurées (sortie IRSNet) en une image reconstruite x_hat via des
convolutions transposées, optimisé pour la fidélité visuelle humaine
(PSNR, SSIM, LPIPS).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Entrée: z_restored (B, d, H, W) avec H=W=feature_map_size (ex H_feat, W_feat).
    Sortie: x_hat (B, 3, H_img, W_img) dans [0, 1].
    Le facteur d'upsampling total doit correspondre au ratio entre
    feature_map_size et image_size (ex backbone stride /8 -> 3 UpBlocks).
    """

    def __init__(self, in_channels, base_channels=128, num_upsamples=3, out_channels=3, min_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        ch = base_channels
        for _ in range(num_upsamples):
            next_ch = max(min_channels, ch // 2)
            blocks.append(UpBlock(ch, next_ch))
            ch = next_ch
        self.upsamples = nn.Sequential(*blocks)
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_restored, target_size=None):
        """
        target_size: (H, W) optionnel. Le nombre d'UpBlocks (donc le facteur
        d'upsampling ×2^num_upsamples) est figé aux poids du réseau à la
        construction -- si la résolution d'entrée réelle varie en production
        (voir net/pipeline.py:_infer_downsample_factor, calculé une seule
        fois à l'init), un écart résiduel peut apparaître entre la sortie
        des UpBlocks et la résolution image attendue. Ce dernier
        interpolate bilinéaire corrige cet écart sans changer l'architecture
        ni recharger les poids -- ne remplace pas un réentraînement pour des
        tailles très différentes de celle utilisée à l'entraînement, mais
        garantit qu'aucun mismatch de forme ne fait planter le pipeline.
        """
        x = self.stem(z_restored)
        x = self.upsamples(x)
        x_hat = self.out_conv(x)
        if target_size is not None and x_hat.shape[-2:] != tuple(target_size):
            x_hat = F.interpolate(x_hat, size=target_size, mode='bilinear', align_corners=False)
        return x_hat
