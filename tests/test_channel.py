"""
Vérifie net/channel.py :
- différentiabilité de bout en bout (pas de porte binaire h_a bloquante)
- device dérivé dynamiquement de l'entrée (pas d'attribut figé -- bug GPU corrigé)
- tendance physique du BER (croît avec la distance, régression sur deux bugs
  réels trouvés : pointing_error_rayleigh sans échelle physique, R_L/G non calibrés)
"""
import torch
from net.channel import Channel


class DummyConfig:
    def __init__(self):
        self.device = torch.device('cpu')
        self.channel_type = 'uwoc_im_dd'
        self.link_distance = 10.0
        self.c_e = 0.151
        self.sigma_T2 = 0.01
        self.D_r = 0.075
        self.w_Leq = 0.05
        self.sigma_theta_t = 2e-3
        self.R_e = 0.8
        self.R_L = 1e3
        self.G = 1e5
        self.B_e = 1e6
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
    """
    `Channel.device` est un attribut Python figé à la construction, PAS un
    buffer nn.Module -- ne suit pas un `.to(cuda)` ultérieur. Vérifie que
    `forward()` dérive le device dynamiquement de `bits.device`, pas de cet
    attribut (on le corrompt volontairement : si forward() en dépendait
    encore, ce test planterait).
    """
    config = DummyConfig()
    channel = Channel(config)
    channel.device = 'un_device_invalide_qui_ne_doit_jamais_etre_utilise'

    bits = torch.rand(1, 4, 4, 4)
    b_hat, p = channel(bits, link_distance=config.link_distance)
    assert b_hat.device == bits.device
    print('test_channel_device_derived_from_input_not_stale_attribute OK')


def test_ber_increases_monotonically_with_distance():
    """
    Régression physique : le BER théorique (formule erfc exacte, §3.2) doit
    croître avec la distance. Détecte deux bugs réels :
    1. pointing_error_rayleigh sans échelle physique (sigma_theta_t manquant)
       -> h≈0 quasi systématiquement, BER≈0.5 même à 1m.
    2. R_L/G mal calibrés -> bruit de grenaille (∝R_L²×G) dominant le signal
       à toute distance.
    """
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


if __name__ == '__main__':
    test_channel_gradient_flows()
    test_channel_device_derived_from_input_not_stale_attribute()
    test_ber_increases_monotonically_with_distance()
