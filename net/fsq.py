"""
Quantification Scalaire Finie Bornée (FSQ) avec Straight-Through Estimator (§3.1).
Conforme à Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple",
arXiv:2309.15505 (Google DeepMind, 2023, Appendice A.1).

z_tilde_i = round( (L_i - 1)*(1 - eps)/2 * tanh(z_i + shift_i) ) - offset_i
où offset_i = 0.5 si L_i est pair (0.0 si impair), et shift_i = atanh(offset_i / half_l_i) (0 pour L=2).

Pour tout L_i >= 2 (pair ou impair) :
- Le nombre de niveaux effectifs est exactement L_i (pas de niveau perdu pour L pair).
- Les valeurs quantifiées sont des entiers dans [-(L_i // 2), (L_i - 1) - (L_i // 2)].
- En ajoutant L_i // 2, on obtient exactement {0, 1, ..., L_i - 1}.
"""
import math
import torch
import torch.nn as nn


class FSQ(nn.Module):
    """
    Finite Scalar Quantization (Mentzer et al., arXiv:2309.15505).
    """

    def __init__(self, levels: list):
        """
        levels: liste des L_i par dimension, ex [8, 8, 8, 6, 5] (config 14 bits).
        """
        super().__init__()
        assert len(levels) <= 8, "d doit être ≤ 8"
        self.levels_list = [int(l) for l in levels]
        self.d = len(self.levels_list)

        levels_t = torch.tensor(self.levels_list, dtype=torch.float32)
        self.register_buffer('levels', levels_t)

        eps = 1e-3
        half_l = (levels_t - 1.0) * (1.0 - eps) / 2.0
        offset = torch.where(levels_t % 2 == 1, 0.0, 0.5)
        # Conforme à Mentzer et al. (arXiv:2309.15505) :
        # - L impair : offset=0, shift=0 (centré naturellement)
        # - L=2 : shift=0 pour placer le seuil de décision exactement à z=0 (50%/50% sous loi normale)
        # - L pair >= 4 : shift = atanh(offset / half_l) pour centrer le niveau zéro
        shift = torch.where(
            levels_t == 2,
            torch.zeros_like(half_l),
            torch.where(
                levels_t % 2 == 1,
                torch.zeros_like(half_l),
                torch.atanh(torch.clamp(offset / half_l, max=0.9999))
            )
        )
        half_width = torch.floor(levels_t / 2.0)

        # Buffers 4D pour le broadcasting vectorisé direct (1, d, 1, 1)
        self.register_buffer('half_l', half_l.view(1, -1, 1, 1))
        self.register_buffer('offset', offset.view(1, -1, 1, 1))
        self.register_buffer('shift', shift.view(1, -1, 1, 1))
        self.register_buffer('half_width', half_width.view(1, -1, 1, 1))

        # Bornes entières exactes
        min_bounds = -half_width
        max_bounds = (levels_t - 1.0) - half_width
        self.register_buffer('min_bounds', min_bounds.view(1, -1, 1, 1))
        self.register_buffer('max_bounds', max_bounds.view(1, -1, 1, 1))

    @property
    def bounds(self):
        """Borne maximale absolue pour compatibilité arrière."""
        return self.half_width.squeeze()

    def forward(self, z):
        """
        z: (B, d, H, W) — sortie du goulot Edge.
        Retourne z_q avec des valeurs entières dans [min_bounds, max_bounds] via STE.
        """
        z_bounded = torch.tanh(z + self.shift) * self.half_l - self.offset
        z_rounded = torch.round(z_bounded)
        # Straight-Through Estimator : forward = valeur discrète, backward = identité
        z_q = z_bounded + (z_rounded - z_bounded).detach()
        return z_q

    def codebook_size(self):
        """Taille du dictionnaire implicite prod(L_i)."""
        return math.prod(self.levels_list)

    def bits_per_token(self):
        """Nombre de bits physiques requis par token : sum(ceil(log2(L_i)))."""
        return sum(math.ceil(math.log2(l)) for l in self.levels_list)

    def information_bits(self):
        """Information théorique en bits : log2(prod(L_i))."""
        return math.log2(self.codebook_size())
