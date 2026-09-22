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
    Retourne z_clamped avec chaque dimension i bornée à [-(L_i-1)/2, +(L_i-1)/2].
    """
    bounds = torch.tensor([(l - 1) / 2.0 for l in levels],
                           dtype=z_hat.dtype, device=z_hat.device).view(1, -1, 1, 1)
    return torch.clamp(z_hat, min=-bounds, max=bounds)
