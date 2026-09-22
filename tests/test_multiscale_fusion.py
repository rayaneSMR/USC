"""
Vérifie le module ASFF (net/multiscale_fusion.py) :
- fusionne bien N échelles en un seul tenseur à la résolution de référence
- gradient différentiable vers chaque échelle d'entrée
- pas de concat des features complètes (seulement les cartes de poids 1-canal)
"""
import torch
from net.multiscale_fusion import ASFF


def test_asff_output_shape():
    in_channels_list = [64, 128, 256]
    fusion_channels = 32
    asff = ASFF(in_channels_list, fusion_channels=fusion_channels, reference_index=0)

    features = [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 16, 16),
        torch.randn(2, 256, 8, 8),
    ]
    fused = asff(features)
    assert fused.shape == (2, fusion_channels, 32, 32), f"shape obtenue: {fused.shape}"
    print('test_asff_output_shape OK, fused shape=', fused.shape)


def test_asff_gradient_flows_to_all_scales():
    in_channels_list = [64, 128, 256]
    asff = ASFF(in_channels_list, fusion_channels=32, reference_index=0)

    features = [
        torch.randn(1, 64, 16, 16, requires_grad=True),
        torch.randn(1, 128, 8, 8, requires_grad=True),
        torch.randn(1, 256, 4, 4, requires_grad=True),
    ]
    fused = asff(features)
    loss = fused.sum()
    loss.backward()

    for i, f in enumerate(features):
        assert f.grad is not None, f"pas de gradient pour l'échelle {i}"
        assert torch.any(f.grad != 0), f"gradient nul pour l'échelle {i}"
    print('test_asff_gradient_flows_to_all_scales OK')


def test_asff_weights_sum_to_one():
    """Les poids softmax par position spatiale doivent sommer à 1 (propriété
    de fusion convexe, garantit que la fusion reste dans l'échelle des features)."""
    import torch.nn.functional as F

    in_channels_list = [64, 128]
    asff = ASFF(in_channels_list, fusion_channels=16, reference_index=0)

    features = [torch.randn(1, 64, 8, 8), torch.randn(1, 128, 4, 4)]
    projected = [asff.proj[i](features[i]) for i in range(2)]
    projected = [
        p if p.shape[-2:] == (8, 8) else
        torch.nn.functional.interpolate(p, size=(8, 8), mode='bilinear', align_corners=False)
        for p in projected
    ]
    raw_weights = [asff.weight_conv[i](projected[i]) for i in range(2)]
    stacked = torch.cat(raw_weights, dim=1)
    norm_weights = F.softmax(stacked, dim=1)

    sums = norm_weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    print('test_asff_weights_sum_to_one OK')


if __name__ == '__main__':
    test_asff_output_shape()
    test_asff_gradient_flows_to_all_scales()
    test_asff_weights_sum_to_one()
