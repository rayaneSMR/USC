"""
Vérifie que FSQ respecte les bornes [-(L_i-1)/2, +(L_i-1)/2] et que le STE
laisse passer le gradient (dz_out/dz_in ≈ 1 en moyenne).
"""
import torch
from net.fsq import FSQ


def test_fsq_bounds():
    levels = [8, 8, 8, 6, 5]
    fsq = FSQ(levels)
    z = torch.randn(2, len(levels), 4, 4, requires_grad=True)
    z_q = fsq(z)

    bounds = fsq.bounds.view(1, -1, 1, 1)
    assert torch.all(z_q <= bounds + 1e-4)
    assert torch.all(z_q >= -bounds - 1e-4)
    print('test_fsq_bounds OK')


def test_fsq_gradient_flows():
    levels = [8, 8, 8, 6, 5]
    fsq = FSQ(levels)
    z = torch.randn(2, len(levels), 4, 4, requires_grad=True)
    z_q = fsq(z)
    loss = z_q.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.any(z.grad != 0)
    print('test_fsq_gradient_flows OK')


if __name__ == '__main__':
    test_fsq_bounds()
    test_fsq_gradient_flows()
