"""
Vérifie net/channel.py :
- différentiabilité de bout en bout (pas de porte binaire h_a bloquante)
- device dérivé dynamiquement de l'entrée (pas d'attribut figé -- bug GPU corrigé)
- tendance physique du BER (croît avec la distance)
- Équation 24 : calcul dynamique de w_Leq à partir de w_L(L) et D_r (pas de constante codée en dur)
- Liaison dynamique de la bande passante B_e ≈ R_b / 2
- Variance de turbulence avec moyennage d'ouverture sigma_T2 ~ 1.1e-7 (Eq. 43)
"""
import math
import torch
from net.channel import Channel


class DummyConfig:
    def __init__(self):
        self.device = torch.device('cpu')
        self.channel_type = 'uwoc_im_dd'
        self.link_distance = 10.0
        self.c_e = 0.151
        self.sigma_T2 = 1.1e-7
        self.D_r = 0.075
        self.theta_div = 0.05
        self.w0 = 20e-3 * math.sqrt(2)
        self.sigma_theta_t = 2e-3
        self.R_b = 1e6
        self.B_e = 0.5e6
        self.R_e = 0.8
        self.R_L = 1e3
        self.G = 1e5
        self.T_e = 290.0
        self.logger = None


def test_channel_gradient_flows():
    config = DummyConfig()
    channel = Channel(config)
    bits = torch.rand(2, 8, 8, 8, requires_grad=True)

    b_hat, p = channel(bits, link_distance=config.link_distance)
    loss = b_hat.sum()
    loss.backward()

    assert bits.grad is not None
    print(f'test_channel_gradient_flows OK (BER={p.item():.4f})')


def test_channel_device_derived_from_input_not_stale_attribute():
    config = DummyConfig()
    channel = Channel(config)
    channel.device = 'un_device_invalide_qui_ne_doit_jamais_etre_utilise'

    bits = torch.rand(1, 4, 4, 4)
    b_hat, p = channel(bits, link_distance=config.link_distance)
    assert b_hat.device == bits.device
    print('test_channel_device_derived_from_input_not_stale_attribute OK')


def test_ber_increases_monotonically_with_distance():
    config = DummyConfig()
    channel = Channel(config)
    bits = torch.randint(0, 2, (1, 4, 16, 16)).float()

    distances = [1.0, 20.0, 60.0]
    bers = []
    for L in distances:
        torch.manual_seed(0)
        _, p = channel(bits, link_distance=L)
        bers.append(p.item())

    assert bers[0] < bers[1] < bers[2], (
        f"BER devrait croître avec la distance, obtenu {list(zip(distances, bers))}"
    )
    assert bers[0] < 0.1, f"BER à 1m anormalement élevé ({bers[0]:.4f})"
    print(f'test_ber_increases_monotonically_with_distance OK, BER={list(zip(distances, bers))}')


def test_w_Leq_equation24_computed_dynamically():
    """
    Vérifie que w_Leq n'est plus codé en dur à 0.05 m, mais calculé
    dynamiquement selon l'Équation 24 à partir de w_L(L) et D_r.
    """
    config = DummyConfig()
    channel = Channel(config)

    # À L = 10m
    A0_10, w_Leq2_10, w_Leq_10 = channel.compute_pe_geometry(10.0, device='cpu')
    w_L_10 = channel.compute_w_L(10.0)

    # À L = 80m
    A0_80, w_Leq2_80, w_Leq_80 = channel.compute_pe_geometry(80.0, device='cpu')
    w_L_80 = channel.compute_w_L(80.0)

    # w_L doit croître avec la distance
    assert w_L_80 > w_L_10, f"w_L(80m)={w_L_80} devrait être > w_L(10m)={w_L_10}"

    # w_Leq doit être différent selon la distance (pas une constante figée)
    assert not math.isclose(w_Leq_10.item(), w_Leq_80.item(), rel_tol=1e-3), (
        f"w_Leq devrait varier avec la distance: {w_Leq_10.item()} vs {w_Leq_80.item()}"
    )
    assert not math.isclose(w_Leq_10.item(), 0.05, abs_tol=1e-4), "w_Leq ne doit pas être figé à 0.05"

    # Vérification exacte de l'équation 24:
    nu = (config.D_r * math.sqrt(math.pi)) / (2.0 * math.sqrt(2.0) * w_L_80)
    expected_w_Leq2 = (w_L_80 ** 2) * (math.sqrt(math.pi) * math.erf(nu)) / (2.0 * nu * math.exp(-nu ** 2))
    assert math.isclose(w_Leq2_80.item(), expected_w_Leq2, rel_tol=1e-5), (
        f"w_Leq^2 ({w_Leq2_80.item()}) ne correspond pas à l'Équation 24 ({expected_w_Leq2})"
    )
    print(f'test_w_Leq_equation24_computed_dynamically OK (w_Leq(10m)={w_Leq_10.item():.4f}m, w_Leq(80m)={w_Leq_80.item():.4f}m)')


def test_bandwidth_dynamically_bound_to_bitrate():
    """
    Vérifie que la bande passante B_e est liée dynamiquement à R_b (B_e ≈ R_b / 2).
    """
    config = DummyConfig()
    channel = Channel(config)

    assert math.isclose(channel.B_e, 0.5e6, rel_tol=1e-5), f"B_e devrait être 0.5e6 pour R_b=1e6, obtenu {channel.B_e}"

    # Changement dynamique de R_b
    channel.set_bitrate(2e6)
    assert math.isclose(channel.B_e, 1.0e6, rel_tol=1e-5), f"B_e devrait être 1.0e6 pour R_b=2e6, obtenu {channel.B_e}"

    channel.set_bitrate(10e6)
    assert math.isclose(channel.B_e, 5.0e6, rel_tol=1e-5), f"B_e devrait être 5.0e6 pour R_b=10e6, obtenu {channel.B_e}"
    print('test_bandwidth_dynamically_bound_to_bitrate OK')


def test_aperture_averaging_turbulence_variance():
    """
    Vérifie que sigma_T2 est dans la plage réaliste ~1.1e-7 avec moyennage d'ouverture.
    """
    config = DummyConfig()
    channel = Channel(config)
    assert math.isclose(channel.sigma_T2, 1.1e-7, rel_tol=1e-4), f"sigma_T2 devrait être 1.1e-7, obtenu {channel.sigma_T2}"

    # Calcul Eq. 43
    sT2_eq43 = channel.compute_sigma_T2_eq43(80.0, 0.075)
    assert math.isclose(sT2_eq43, 1.1e-7, rel_tol=1e-3), f"Eq. 43 devrait donner ~1.1e-7 à 80m, obtenu {sT2_eq43}"
    print(f'test_aperture_averaging_turbulence_variance OK (sigma_T2={channel.sigma_T2:.2e}, Eq43={sT2_eq43:.2e})')


if __name__ == '__main__':
    test_channel_gradient_flows()
    test_channel_device_derived_from_input_not_stale_attribute()
    test_ber_increases_monotonically_with_distance()
    test_w_Leq_equation24_computed_dynamically()
    test_bandwidth_dynamically_bound_to_bitrate()
    test_aperture_averaging_turbulence_variance()
