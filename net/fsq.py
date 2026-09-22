"""
Quantification Scalaire Finie Bornée (FSQ) avec Straight-Through Estimator (§3.1).

z_tilde_i = round( (L_i - 1)/2 * tanh(z_i) ),  z_tilde_i in {-(L_i-1)/2, ..., +(L_i-1)/2}

Dictionnaire implicite = produit cartésien des niveaux (prod L_i), sans
commitment loss ni codebook explicite -> pas de codebook collapse.
Confinement d'erreur intrinsèque : une inversion binaire ne perturbe qu'une
seule valeur scalaire locale bornée.
"""
import torch
import torch.nn as nn


class FSQ(nn.Module):
    def __init__(self, levels: list):
        """
        levels: liste des L_i par dimension, ex [8, 8, 8, 6, 5] (config 14 bits).
        """
        super().__init__()
        assert len(levels) <= 8, "d doit être ≤ 8"
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.float32))
        self.d = len(levels)

    @property
    def bounds(self):
        return (self.levels - 1) / 2.0

    def forward(self, z):
        """
        z: (B, d, H, W) — sortie du goulot Edge.
        Retourne z_q dans les bornes [-(L_i-1)/2, +(L_i-1)/2], différentiable (STE).
        """
        bound = self.bounds.view(1, -1, 1, 1).to(z.device)
        z_bounded = torch.tanh(z) * bound
        z_rounded = torch.round(z_bounded)
        # Straight-Through Estimator : forward = valeur discrète, backward = identité
        z_q = z_bounded + (z_rounded - z_bounded).detach()
        return z_q

    def codebook_size(self):
        size = 1
        for l in self.levels.tolist():
            size *= int(l)
        return size

    def bits_per_token(self):
        import math
        return sum(math.ceil(math.log2(int(l))) for l in self.levels.tolist())
