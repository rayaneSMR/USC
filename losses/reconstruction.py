"""
Pertes de reconstruction visuelle (Phase 1, 2, 3) : L1 + métriques PSNR/SSIM/LPIPS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def l1_loss(x, x_hat):
    """L_P1 = ||x - x_hat||_1"""
    return F.l1_loss(x_hat, x)


def psnr(x, x_hat, max_val=1.0):
    mse = F.mse_loss(x_hat, x)
    if mse.item() == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(max_val ** 2 / mse)


def ssim(x, x_hat, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Implémentation SSIM simplifiée (fenêtre gaussienne moyenne)."""
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_xh = F.avg_pool2d(x_hat, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
    sigma_xh = F.avg_pool2d(x_hat * x_hat, window_size, stride=1, padding=window_size // 2) - mu_xh ** 2
    sigma_xxh = F.avg_pool2d(x * x_hat, window_size, stride=1, padding=window_size // 2) - mu_x * mu_xh

    ssim_map = ((2 * mu_x * mu_xh + C1) * (2 * sigma_xxh + C2)) / (
        (mu_x ** 2 + mu_xh ** 2 + C1) * (sigma_x + sigma_xh + C2)
    )
    return ssim_map.mean()


class LPIPSMetric:
    """Wrapper optionnel autour de la lib `lpips` (nécessite `pip install lpips`)."""

    def __init__(self, net='alex', device='cpu'):
        if not _LPIPS_AVAILABLE:
            self.model = None
        else:
            self.model = lpips_lib.LPIPS(net=net).to(device)

    def __call__(self, x, x_hat):
        if self.model is None:
            return torch.tensor(float('nan'))
        # lpips attend des valeurs dans [-1, 1]
        return self.model(x_hat * 2 - 1, x * 2 - 1).mean()
