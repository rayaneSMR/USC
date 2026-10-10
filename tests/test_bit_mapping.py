"""
Vérifie net/bit_mapping.py :
- encode() puis decode() reconstruit exactement les mêmes valeurs sans bruit
- le gradient traverse encode() ET decode() (bug STE corrigé : le seuillage
  dur `(bits > 0.5)` dans `_bits_to_int` bloquait totalement le gradient
  avant correction).
"""
import torch
from net.bit_mapping import BitMapper
from net.fsq import FSQ


def test_encode_decode_roundtrip_exact():
    levels = [8, 8, 8, 6, 5]
    fsq = FSQ(levels)
    bm = BitMapper(levels)

    # Vérification du budget de bits exact : 3+3+3+3+3 = 15 bits physiques
    assert bm.bits_per_dim == [3, 3, 3, 3, 3], f"Attendu [3, 3, 3, 3, 3], obtenu {bm.bits_per_dim}"
    assert bm.total_bits == 15, f"Attendu 15 bits totaux, obtenu {bm.total_bits}"

    z = torch.randn(2, len(levels), 4, 4) * 3.0
    z_q = fsq(z)
    b = bm.encode(z_q)
    z_hat = bm.decode(b)

    assert torch.allclose(z_q, z_hat, atol=1e-4), "roundtrip encode/decode incorrect sans bruit de canal"
    print('test_encode_decode_roundtrip_exact OK')


def test_5bit_config_bit_budget_and_roundtrip():
    """Vérifie la configuration 5-bit [2, 2, 2, 2, 2] : 5 bits physiques, roundtrip exact."""
    levels = [2, 2, 2, 2, 2]
    fsq = FSQ(levels)
    bm = BitMapper(levels)

    assert bm.bits_per_dim == [1, 1, 1, 1, 1]
    assert bm.total_bits == 5

    z = torch.randn(4, 5, 8, 8) * 3.0
    z_q = fsq(z)
    b = bm.encode(z_q)
    z_hat = bm.decode(b)

    assert torch.allclose(z_q, z_hat, atol=1e-4)
    print('test_5bit_config_bit_budget_and_roundtrip OK')


def test_gradient_flows_through_encode():
    levels = [8, 8, 8, 6, 5]
    bm = BitMapper(levels)
    z_q = torch.randn(1, len(levels), 4, 4, requires_grad=True)
    b = bm.encode(z_q)
    loss = b.sum()
    loss.backward()
    assert z_q.grad is not None
    assert torch.any(z_q.grad != 0)
    print('test_gradient_flows_through_encode OK')


def test_gradient_flows_through_decode():
    """Régression du bug STE : avant correction, `_bits_to_int` utilisait un
    seuillage dur sans STE, ce qui bloquait tout gradient à travers decode()."""
    levels = [8, 8, 8, 6, 5]
    bm = BitMapper(levels)

    b = torch.rand(1, bm.total_bits, 4, 4, requires_grad=True)  # bits "mous" pré-canal
    z_hat = bm.decode(b)
    loss = z_hat.sum()
    loss.backward()
    assert b.grad is not None, "régression du bug STE : gradient bloqué dans decode()"
    assert torch.any(b.grad != 0)
    print('test_gradient_flows_through_decode OK')


def test_gradient_flows_end_to_end_encode_decode():
    levels = [8, 8, 8, 6, 5]
    fsq = FSQ(levels)
    bm = BitMapper(levels)

    z = torch.randn(1, len(levels), 4, 4, requires_grad=True)
    z_q = fsq(z)
    b = bm.encode(z_q)
    z_hat = bm.decode(b)  # pas de canal entre les deux : simule Phase 1 (use_channel=False)
    loss = z_hat.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.any(z.grad != 0)
    print('test_gradient_flows_end_to_end_encode_decode OK')


if __name__ == '__main__':
    test_encode_decode_roundtrip_exact()
    test_5bit_config_bit_budget_and_roundtrip()
    test_gradient_flows_through_encode()
    test_gradient_flows_through_decode()
    test_gradient_flows_end_to_end_encode_decode()
