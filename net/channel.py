"""
Canal physique optique sous-marin différentiable (§3.2), IM/DD NRZ-OOK.

h = h_c * h_t * h_p (Beer-Lambert x turbulence log-normale x pointing error
Rayleigh), sans la variable discrète h_a (portes binaires supprimées pour
garder un canal entièrement différentiable, cf. justification §3.2).
Bruit gaussien post-détection dépendant du signal (grenaille + thermique TIA).
Démodulation à seuil optimal gamma_th -> BER théorique p (formule erfc §3.2).
"""
import math
import torch
import torch.nn as nn


class Channel(nn.Module):
    """chan_type: 'none' | 'uwoc_im_dd'"""

    def __init__(self, args_or_config, config=None):
        super().__init__()
        cfg = config if config is not None else args_or_config
        self.config = cfg
        self.chan_type = getattr(cfg, 'channel_type', 'uwoc_im_dd')
        # NOTE GPU : `self.device` n'est utilisé que pour les logs -- JAMAIS
        # pour créer des tenseurs dans forward()/les méthodes ci-dessous.
        # Un attribut Python (contrairement à un buffer nn.Module) ne suit
        # pas un `.to(cuda)` appelé après la construction (ex checkpoint
        # chargé CPU puis déplacé GPU) -- toutes les méthodes dérivent donc
        # le device dynamiquement de `bits.device` à chaque appel.
        self.device = cfg.device

        self.c_e = cfg.c_e
        self.sigma_T2 = cfg.sigma_T2
        self.D_r = cfg.D_r
        self.w_Leq = cfg.w_Leq
        self.sigma_theta_t = getattr(cfg, 'sigma_theta_t', 2e-3)  # rad, gigue angulaire (§3.2)
        self.R_e = cfg.R_e
        self.R_L = cfg.R_L
        self.G = cfg.G
        self.B_e = cfg.B_e
        self.T_e = cfg.T_e

        if cfg.logger:
            cfg.logger.info('【Channel】: Built {} channel, distance {} m.'.format(
                self.chan_type, cfg.link_distance))

    # ---- composantes du gain de canal h = h_c . h_t . h_p ----

    def beer_lambert_gain(self, L, device):
        """Atténuation déterministe (absorption + diffusion), c_e ~ 0.151 m^-1 (eau claire)."""
        L_t = torch.as_tensor(L, dtype=torch.float32, device=device)
        return torch.exp(-self.c_e * L_t)

    def turbulence_lognormal(self, shape, device):
        """Turbulence océanique log-normale stratifiée, régime de faible turbulence
        (aperture averaging avec lentille D_r), mu_T = -sigma_T^2 / 2."""
        mu_T = -0.5 * self.sigma_T2
        T = torch.normal(mean=mu_T, std=math.sqrt(self.sigma_T2), size=shape, device=device)
        return torch.exp(T)

    def pointing_error_rayleigh(self, shape, device, link_distance):
        """
        Erreur de pointage géométrique (déviation radiale r_d, distribution
        de Rayleigh liée à la gigue angulaire de surface σ_θt -- §3.2).
        r_d = σ_θt (rad) x distance (m) x |N(0,1)| -- approximation small-angle
        standard en optique espace libre (déviation radiale ≈ angle x distance).
        """
        r_d = self.sigma_theta_t * link_distance * torch.abs(
            torch.normal(mean=0.0, std=1.0, size=shape, device=device))
        nu = self.D_r * math.pi / (math.sqrt(2) * self.w_Leq)
        A0 = torch.erf(torch.tensor(nu, device=device)) ** 2
        return A0 * torch.exp(-2 * r_d ** 2 / self.w_Leq ** 2)

    def channel_gain(self, shape, link_distance, device):
        h_c = self.beer_lambert_gain(link_distance, device)
        h_t = self.turbulence_lognormal(shape, device)
        h_p = self.pointing_error_rayleigh(shape, device, link_distance)
        return h_c * h_t * h_p

    # ---- bruit post-détection (grenaille + thermique) ----

    def sipm_noise_std(self, I_s, I_d=0.0, I_b=0.0):
        """sigma_n^2 = R_L^2 . 2e.G.F.B_e.(I_s+I_d+I_b) + 4.k_B.T_e.B_e.R_L"""
        k_B = 1.380649e-23
        e_charge = 1.602176634e-19
        shot = 2 * e_charge * self.G * self.B_e * (I_s + I_d + I_b)
        thermal = 4 * k_B * self.T_e * self.B_e / self.R_L
        sigma_n2 = (self.R_L ** 2) * shot + thermal
        return torch.sqrt(torch.clamp(sigma_n2, min=1e-30))

    def optimal_threshold_ber(self, r0_mean, r1_mean, sigma_n0, sigma_n1):
        """BER théorique conditionnel p, via seuil gaussien optimal gamma_th (§3.2)."""
        gamma_th = (r0_mean * sigma_n1 + r1_mean * sigma_n0) / (sigma_n0 + sigma_n1)
        term0 = torch.erfc((gamma_th - r0_mean) / (math.sqrt(2) * sigma_n0))
        term1 = torch.erfc((r1_mean - gamma_th) / (math.sqrt(2) * sigma_n1))
        p = 0.25 * term0 + 0.25 * term1
        return p, gamma_th

    # ---- interface principale ----

    def forward(self, bits: torch.Tensor, link_distance: float):
        """
        bits: (B, n_bits, H, W) flux binaire {0,1} float, différentiable.
        Retourne (b_hat, p) : bits reçus (STE) et BER théorique moyen (§3.2,
        formule exacte via erfc, calculée par élément puis moyennée).
        """
        device = bits.device

        if self.chan_type == 'none':
            return bits, torch.zeros(1, device=device)

        shape = bits.shape
        h = self.channel_gain(shape, link_distance, device)

        P_tx = 1.0  # puissance optique normalisée (OOK idéal : 0 si bit=0, P_tx si bit=1)
        I_s1 = self.R_e * h * P_tx
        I_s0 = torch.zeros_like(I_s1)

        sigma_n1 = self.sipm_noise_std(I_s1)
        sigma_n0 = self.sipm_noise_std(I_s0)

        # Seuil optimal γ_th et BER théorique EXACTS (§3.2, formule erfc),
        # calculés par élément (dépendent de h, donc varient dans l'image).
        p_elementwise, gamma_th = self.optimal_threshold_ber(I_s0, I_s1, sigma_n0, sigma_n1)

        # Signal effectivement reçu selon le bit RÉELLEMENT envoyé
        I_s = torch.where(bits >= 0.5, I_s1, I_s0)
        sigma_n = torch.where(bits >= 0.5, sigma_n1, sigma_n0)
        r = I_s + torch.randn_like(I_s) * sigma_n

        b_hat_hard = (r > gamma_th).float()

        # STE : forward = décision dure, backward = gradient identité.
        noise = (b_hat_hard - bits).detach()
        b_hat = bits + noise

        p = p_elementwise.mean()

        return b_hat, p
