"""
MATK — attaque de masque différentiable (§4, Phase 2A).
Masque aléatoirement une fraction (ratio = 2p) des positions spatiales du
tenseur latent pour forcer IRSNet + G_psi à exploiter les corrélations
spatiales profondes afin de reconstruire les zones manquantes, en plus du
bruit de canal (BFP).
"""
import torch
import torch.nn as nn


class MATK(nn.Module):
    def __init__(self, ratio_factor: float = 2.0):
        """
        ratio_factor: multiplicateur appliqué au BER courant p pour obtenir
        le ratio de masquage (ratio = ratio_factor * p), cf. §4 Phase 2A.
        """
        super().__init__()
        self.ratio_factor = ratio_factor
        self.enabled = True

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        z: (B, d, H, W) tenseur latent (typiquement après clamp, avant IRSNet).
        p: BER scalaire courant (issu du canal).
        """
        if not self.enabled:
            return z

        ratio = torch.clamp(self.ratio_factor * p, 0.0, 0.9)
        b, d, h, w = z.shape
        # masque partagé sur toutes les dimensions à une position spatiale donnée
        mask = (torch.rand(b, 1, h, w, device=z.device) > ratio).float()
        return z * mask

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
