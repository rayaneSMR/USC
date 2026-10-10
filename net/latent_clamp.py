"""
Écrêtage strict par dimension (§3.3), appliqué au récepteur avant IRSNet,
pour forcer chaque dimension démappée à rester dans ses bornes FSQ
d'entraînement et éviter des activations aberrantes dans le décodeur.
"""
import torch


def clamp_latent(z_hat: torch.Tensor, levels: list) -> torch.Tensor:
    """
    z_hat: (B, d, H, W), issu du démappage binaire (peut être hors-bornes
           si des bits ont été inversés par le canal).
    levels: liste des L_i par dimension.
    Retourne z_clamped avec chaque dimension i bornée à
    [-(L_i // 2), (L_i - 1) - (L_i // 2)], conforme à Mentzer et al. (arXiv:2309.15505).
    """
    if isinstance(levels, torch.Tensor):
        levels_list = levels.tolist()
    else:
        levels_list = levels

    min_bounds = torch.tensor([-(int(l) // 2) for l in levels_list],
                              dtype=z_hat.dtype, device=z_hat.device).view(1, -1, 1, 1)
    max_bounds = torch.tensor([(int(l) - 1) - (int(l) // 2) for l in levels_list],
                              dtype=z_hat.dtype, device=z_hat.device).view(1, -1, 1, 1)
    return torch.clamp(z_hat, min=min_bounds, max=max_bounds)
