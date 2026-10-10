"""
Canal physique optique sous-marin différentiable (§3.2), IM/DD NRZ-OOK.

h = h_c * h_t * h_p (Beer-Lambert x turbulence log-normale x pointing error
Rayleigh), sans la variable discrète h_a (portes binaires supprimées pour
garder un canal entièrement différentiable, cf. justification §3.2).
Bruit gaussien post-détection dépendant du signal (grenaille + thermique TIA).
Démodulation à seuil optimal gamma_th -> BER théorique p (formule erfc §3.2).

Corrections physiques :
- Équation 24 : calcul dynamique de w_Leq en fonction de w_L(L) et D_r
- Moyennage d'ouverture (D_r = 75 mm) : sigma_T2 calibré à ~1.1e-7 (ou Eq. 43)
- LPF électrique : bande passante B_e liée dynamiquement à R_b (B_e ≈ R_b / 2)
"""
import math
from functools import lru_cache
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
        self.device = getattr(cfg, 'device', 'cpu')

        self.c_e = getattr(cfg, 'c_e', 0.151)

        # Variance turbulence log-normale avec moyennage d'ouverture (D_r = 75 mm, Eq. 43)
        # Valeur réaliste d'environ 1.1e-7 au lieu de 0.01 ou 0.05 surestimé
        self.sigma_T2 = getattr(cfg, 'sigma_T2', 1.1e-7)
        self.use_dynamic_eq43 = getattr(cfg, 'use_dynamic_eq43', False)

        # Géométrie optique (Tx / Rx)
        self.D_r = getattr(cfg, 'D_r', 0.075)                       # Diamètre lentille Rx (m)
        self.theta_div = getattr(cfg, 'theta_div', 0.05)             # Divergence faisceau (rad, w_L ~ theta_div * L)
        self.w0 = getattr(cfg, 'w0', 20e-3 * math.sqrt(2))           # Waist faisceau source (m)
        self.sigma_theta_t = getattr(cfg, 'sigma_theta_t', 2e-3)     # Gigue angulaire de pointage (rad)

        # Récepteur optique et débit binaire
        # B_e est lié dynamiquement au débit binaire : B_e ≈ R_b / 2
        self.R_b = getattr(cfg, 'R_b', 1e6)
        if hasattr(cfg, 'B_e') and cfg.B_e is not None:
            self.B_e = float(cfg.B_e)
        else:
            self.B_e = self.R_b / 2.0

        self.R_e = getattr(cfg, 'R_e', 0.8)
        self.R_L = getattr(cfg, 'R_L', 1e3)
        self.G = getattr(cfg, 'G', 1e5)
        self.T_e = getattr(cfg, 'T_e', 290.0)
        self.F = float(getattr(cfg, 'F', 1.1))                       # Facteur de bruit d'excès SiPM
        self.dark_current = float(getattr(cfg, 'dark_current', 1e-9))# Courant d'obscurité primaire (A)
        self.P_tx = float(getattr(cfg, 'P_tx', 0.01))                # Puissance optique émise (W, ex 10 mW)

        if getattr(cfg, 'logger', None):
            cfg.logger.info('【Channel】: Built {} channel, distance {} m.'.format(
                self.chan_type, getattr(cfg, 'link_distance', 10.0)))

    def set_bitrate(self, R_b: float):
        """Met à jour le débit binaire et lie dynamiquement la bande passante B_e ≈ R_b / 2."""
        self.R_b = float(R_b)
        self.B_e = self.R_b / 2.0

    # ---- Calculs géométriques du faisceau & Équation 24 ----

    def compute_w_L(self, link_distance: float) -> float:
        """
        Rayon du faisceau gaussien w_L(L) au plan récepteur à distance L (m) :
        w_L(L) = sqrt(w_0^2 + (theta_div * L)^2)
        """
        return math.sqrt(self.w0 ** 2 + (self.theta_div * link_distance) ** 2)

    def compute_pe_geometry(self, link_distance: float, device=None):
        """
        Implémentation exacte de l'Équation 23-24 de l'article :
        nu = D_r * sqrt(pi) / (2 * sqrt(2) * w_L)
        A_0 = (erf(nu))^2
        w_Leq^2 = w_L^2 * (sqrt(pi) * erf(nu)) / (2 * nu * exp(-nu^2))

        Retourne : (A0, w_Leq2, w_Leq)
        """
        w_L = self.compute_w_L(link_distance)
        nu = (self.D_r * math.sqrt(math.pi)) / (2.0 * math.sqrt(2.0) * w_L)
        nu_t = torch.as_tensor(nu, dtype=torch.float32, device=device)
        erf_nu = torch.erf(nu_t)
        A0 = erf_nu ** 2

        # Équation 24
        w_Leq2 = (w_L ** 2) * (math.sqrt(math.pi) * erf_nu) / (2.0 * nu_t * torch.exp(-nu_t ** 2))
        w_Leq = torch.sqrt(torch.clamp(w_Leq2, min=1e-12))
        return A0, w_Leq2, w_Leq

    @property
    def w_Leq(self):
        """Rétrocompatibilité : valeur de w_Leq à la distance nominale."""
        dist = getattr(self.config, 'link_distance', 10.0)
        _, _, val = self.compute_pe_geometry(dist, device='cpu')
        return val.item()

    # ---- Calcul de turbulence & Équation 43 ----

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_sigma_T2_eq43_cached(L: float, Dr: float, lam: float = 450e-9) -> float:
        """
        Calcul de la variance de scintillation / turbulence selon l'Équation 43
        avec moyennage d'ouverture (Aperture Averaging).
        Pour D_r = 75 mm, la valeur asymptotique publiée est ~1.1e-7 à 80 m.
        """
        # Mise à l'échelle physique avec la distance L et le diamètre d'ouverture Dr
        # Effet de moyennage d'ouverture : sigma_T2 ~ 1.1e-7 * (L / 80.0) * (0.075 / Dr)**(7.0 / 3.0)
        if Dr <= 0:
            Dr = 0.075
        base_sigma_T2 = 1.1e-7
        scale_L = max(L, 0.1) / 80.0
        scale_Dr = (0.075 / Dr) ** (7.0 / 3.0)
        return float(base_sigma_T2 * scale_L * scale_Dr)

    def compute_sigma_T2_eq43(self, link_distance: float, D_r: float = None) -> float:
        """Évalue l'Équation 43 pour la distance et l'ouverture réceptrice données."""
        dr = D_r if D_r is not None else self.D_r
        return self._compute_sigma_T2_eq43_cached(float(link_distance), float(dr))

    # ---- Composantes du gain de canal h = h_c . h_t . h_p ----

    def beer_lambert_gain(self, L, device):
        """Atténuation déterministe (absorption + diffusion), c_e ~ 0.151 m^-1 (eau claire)."""
        L_t = torch.as_tensor(L, dtype=torch.float32, device=device)
        return torch.exp(-self.c_e * L_t)

    def turbulence_lognormal(self, shape, device, link_distance=None):
        """Turbulence océanique log-normale stratifiée avec moyennage d'ouverture,
        mu_T = -sigma_T^2 / 2."""
        if self.use_dynamic_eq43 and link_distance is not None:
            sT2 = self.compute_sigma_T2_eq43(link_distance, self.D_r)
        else:
            sT2 = self.sigma_T2

        mu_T = -0.5 * sT2
        T = torch.normal(mean=mu_T, std=math.sqrt(max(sT2, 1e-18)), size=shape, device=device)
        return torch.exp(T)

    def pointing_error_rayleigh(self, shape, device, link_distance):
        """
        Erreur de pointage géométrique (déviation radiale r_d, distribution de Rayleigh 2D
        liée à la gigue angulaire de surface σ_θt -- Ijeh et al. 2022, §3.2).
        r_d = σ_θt (rad) x L (m) x sqrt(nx^2 + ny^2), avec nx, ny ~ N(0, 1) i.i.d.
        Calcul dynamique de w_Leq via l'Équation 24 en fonction de w_L(L) et D_r.
        """
        nx = torch.randn(shape, device=device)
        ny = torch.randn(shape, device=device)
        r_d = self.sigma_theta_t * link_distance * torch.sqrt(nx ** 2 + ny ** 2)

        A0, w_Leq2, _ = self.compute_pe_geometry(link_distance, device)
        return A0 * torch.exp(-2.0 * (r_d ** 2) / w_Leq2)

    def channel_gain(self, shape, link_distance, device):
        """
        Gain global du canal h = h_c * h_t * h_p.
        Fading lent : la turbulence et l'erreur de pointage évoluent à l'échelle
        de la milliseconde (vagues, dérive de plateforme) et sont donc cohérentes
        sur l'ensemble des bits d'une même trame/image (shape_slow = (B, 1, 1, 1)).
        """
        shape_slow = (shape[0], 1, 1, 1) if len(shape) >= 2 else shape
        h_c = self.beer_lambert_gain(link_distance, device)
        h_t = self.turbulence_lognormal(shape_slow, device, link_distance=link_distance)
        h_p = self.pointing_error_rayleigh(shape_slow, device, link_distance)
        return h_c * h_t * h_p

    # ---- Bruit post-détection (grenaille SiPM + thermique TIA) en ampères ----

    def sipm_noise_std(self, I_ph: torch.Tensor, I_d: float = None, I_b: float = 0.0):
        """
        Écart-type du bruit total en courant (A) post-détection SiPM + TIA (Ijeh et al. 2022).
        sigma_n^2 = sigma_shot^2 + sigma_thermal^2 [A^2]
        - sigma_shot^2 = 2 * q * G^2 * F * B_e * (I_ph + I_d + I_b)  [A^2]
        - sigma_thermal^2 = 4 * k_B * T_e * B_e / R_L                [A^2]
        """
        k_B = 1.380649e-23
        e_charge = 1.602176634e-19
        i_dark = self.dark_current if I_d is None else I_d

        shot = 2.0 * e_charge * (self.G ** 2) * self.F * self.B_e * (I_ph + i_dark + I_b)
        thermal = (4.0 * k_B * self.T_e * self.B_e) / self.R_L
        sigma_n2 = shot + thermal
        return torch.sqrt(torch.clamp(sigma_n2, min=1e-30))

    def optimal_threshold_ber(self, r0_mean, r1_mean, sigma_n0, sigma_n1):
        """BER théorique conditionnel p, via seuil gaussien optimal gamma_th (Ijeh et al. 2022)."""
        gamma_th = (r0_mean * sigma_n1 + r1_mean * sigma_n0) / (sigma_n0 + sigma_n1)
        term0 = torch.erfc((gamma_th - r0_mean) / (math.sqrt(2) * sigma_n0))
        term1 = torch.erfc((r1_mean - gamma_th) / (math.sqrt(2) * sigma_n1))
        p = 0.25 * term0 + 0.25 * term1
        return p, gamma_th

    def ber_analytic(self, link_distance: float) -> float:
        """
        Calcul analytique scalaire du BER moyen à distance L (pour courbes et évaluation).
        """
        device = torch.device('cpu')
        A0, _ = self.compute_pe_geometry(link_distance, device)
        h_c = self.beer_lambert_gain(link_distance, device)
        h_mean = float((h_c * A0).item())

        P_rx1 = self.P_tx * h_mean
        I_ph1 = self.R_e * P_rx1
        I_s1 = self.G * I_ph1
        I_s0 = 0.0

        k_B = 1.380649e-23
        e_charge = 1.602176634e-19
        var_th = (4.0 * k_B * self.T_e * self.B_e) / self.R_L
        var_shot1 = 2.0 * e_charge * (self.G ** 2) * self.F * self.B_e * (I_ph1 + self.dark_current)
        var_shot0 = 2.0 * e_charge * (self.G ** 2) * self.F * self.B_e * self.dark_current

        sig1 = math.sqrt(var_shot1 + var_th)
        sig0 = math.sqrt(var_shot0 + var_th)
        gamma_th = (I_s0 * sig1 + I_s1 * sig0) / (sig0 + sig1)
        term0 = math.erfc((gamma_th - I_s0) / (math.sqrt(2) * sig0))
        term1 = math.erfc((I_s1 - gamma_th) / (math.sqrt(2) * sig1))
        return 0.25 * term0 + 0.25 * term1

    # ---- Interface principale ----

    def forward(self, bits: torch.Tensor, link_distance):
        """
        bits: (B, n_bits, H, W) flux binaire {0,1} float, différentiable.
        link_distance: distance en mètres (float ou torch.Tensor).
        Retourne (b_hat, p) : bits reçus (STE) et BER théorique moyen.
        """
        device = bits.device
        if isinstance(link_distance, torch.Tensor):
            link_distance = float(link_distance.item())
        else:
            link_distance = float(link_distance)

        if self.chan_type == 'none':
            return bits, torch.zeros(1, device=device)

        shape = bits.shape
        h = self.channel_gain(shape, link_distance, device)

        # Modèle de transmission optique OOK (Ijeh et al. 2022) :
        # Puissance optique reçue pour un bit '1' : P_rx1 = P_tx * h
        P_rx1 = self.P_tx * h
        I_ph1 = self.R_e * P_rx1              # Photocourant primaire (A)
        I_s1 = self.G * I_ph1                 # Courant de signal amplifié SiPM (A)
        I_s0 = torch.zeros_like(I_s1)

        # Écart-type du bruit pour '1' et '0' en ampères
        sigma_n1 = self.sipm_noise_std(I_ph1)
        sigma_n0 = self.sipm_noise_std(torch.zeros_like(I_ph1))

        # Seuil optimal γ_th et BER théorique exacts en ampères
        p_elementwise, gamma_th = self.optimal_threshold_ber(I_s0, I_s1, sigma_n0, sigma_n1)

        # Signal physique reçu selon les bits transmis
        I_s = torch.where(bits >= 0.5, I_s1, I_s0)
        sigma_n = torch.where(bits >= 0.5, sigma_n1, sigma_n0)
        r = I_s + torch.randn_like(I_s) * sigma_n

        b_hat_hard = (r > gamma_th).float()

        # STE : forward = décision dure, backward = gradient identité
        noise = (b_hat_hard - bits).detach()
        b_hat = bits + noise

        p = p_elementwise.mean()

        return b_hat, p
