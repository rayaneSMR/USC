"""
Vérifie que FSQ respecte les bornes exactes [-(L_i // 2), (L_i - 1) - (L_i // 2)]
selon Mentzer et al. (arXiv:2309.15505), produit exactement L_i niveaux discrets
sans perte pour L pair et pour L=2, et que le STE laisse passer le gradient.
"""
import torch
from net.fsq import FSQ


def test_fsq_bounds_and_levels():
    levels = [8, 8, 8, 6, 5]
    fsq = FSQ(levels)
    z = torch.randn(200, len(levels), 16, 16, requires_grad=True) * 5.0
    z_q = fsq(z)

    assert torch.all(z_q <= fsq.max_bounds + 1e-4)
    assert torch.all(z_q >= fsq.min_bounds - 1e-4)

    # Vérifie qu'aucun niveau n'est perdu (exactement L_i niveaux atteints)
    for i, L in enumerate(levels):
        unique_vals = torch.unique(z_q[:, i])
        assert len(unique_vals) == L, f"Attendu {L} niveaux pour dimension {i}, obtenu {len(unique_vals)}"
    print('test_fsq_bounds_and_levels OK')


def test_fsq_5bit_reaches_both_levels():
    """Régression critique : vérifie que L=2 ne produit pas un tenseur nul."""
    levels = [2, 2, 2, 2, 2]
    fsq = FSQ(levels)
    z = torch.randn(100, 5, 8, 8) * 4.0
    z_q = fsq(z)

    for i in range(5):
        unique_vals = torch.unique(z_q[:, i])
        assert len(unique_vals) == 2, f"L=2 doit produire 2 niveaux {-1, 0}, obtenu {len(unique_vals)}"
        assert not torch.all(z_q[:, i] == 0), "L=2 ne doit pas être identiquement nul !"
    print('test_fsq_5bit_reaches_both_levels OK')


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
    test_fsq_bounds_and_levels()
    test_fsq_5bit_reaches_both_levels()
    test_fsq_gradient_flows()
