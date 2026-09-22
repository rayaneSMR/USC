"""
Vérifie la forme de sortie d'IRSNet (doit préserver H, W et le nombre de canaux
d, cf. §3.3).
"""
import torch
from net.irsnet import IRSNet


def test_irsnet_shape():
    d = 5
    irsnet = IRSNet(in_channels=d)
    z = torch.randn(2, d, 32, 32)
    out = irsnet(z)
    assert out.shape == z.shape, f'Attendu {z.shape}, obtenu {out.shape}'
    print('test_irsnet_shape OK')


if __name__ == '__main__':
    test_irsnet_shape()
