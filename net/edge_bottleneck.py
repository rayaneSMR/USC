"""
Goulot d'étranglement Edge (§3.1) : attention Squeeze-and-Excitation +
convolution 1x1 pour réduire le nombre de canaux à d ≤ 8, avant quantification FSQ.
"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation : pondère dynamiquement les canaux pertinents
    (textures/contours) et atténue le bruit de fond marin."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EdgeBottleneck(nn.Module):
    """
    SE(x) -> Conv1x1(C_in -> d), d ≤ 8.
    Sortie: tenseur (B, d, H, W) prêt pour la quantification FSQ.
    """

    def __init__(self, in_channels, bottleneck_dim=5, se_reduction=16):
        super().__init__()
        assert bottleneck_dim <= 8, "d doit être ≤ 8 (contrainte edge, §3.1)"
        self.se = SEBlock(in_channels, reduction=se_reduction)
        self.proj = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1)

    def forward(self, x):
        x = self.se(x)
        z = self.proj(x)
        return z  # (B, d, H, W), pré-FSQ
