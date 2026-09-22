"""
Mappage binaire par dimension (§2, flux binaire b) et démappage inverse côté
récepteur. Chaque dimension i a L_i niveaux, pas nécessairement une puissance
de 2 -> nombre de bits variable par dimension = ceil(log2(L_i)).
"""
import math
import torch
import torch.nn as nn


class BitMapper(nn.Module):
    def __init__(self, levels: list):
        super().__init__()
        self.levels = levels

        # IMPORTANT : FSQ (net/fsq.py) produit des valeurs de z_q toujours
        # ENTIÈRES via torch.round(bound * tanh(z)), quel que soit L_i. Or la
        # borne théorique (L_i-1)/2 est un DEMI-ENTIER quand L_i est pair
        # (ex L=8 -> bound=3.5). Utiliser ce décalage fractionnaire pour
        # calculer l'index binaire introduit un arrondi supplémentaire
        # (bankers' rounding sur les valeurs .5) qui perd 0.5 d'information
        # à chaque fois -- bug de roundtrip découvert et corrigé ici.
        #
        # Fix : décaler par l'entier le plus proche de la borne
        # (`round(bound)`), pas par la borne fractionnaire elle-même. Cela
        # garantit un alignement exact avec les valeurs entières réellement
        # produites par FSQ, quitte à couvrir une plage d'indices légèrement
        # plus large que L_i pour les niveaux pairs (ex 9 valeurs au lieu de
        # 8 pour L=8) -- le nombre de bits est dimensionné en conséquence.
        self._int_offset = [round((l - 1) / 2.0) for l in levels]
        self._range_size = [2 * off + 1 for off in self._int_offset]
        self.bits_per_dim = [max(1, math.ceil(math.log2(r))) for r in self._range_size]
        self.total_bits = sum(self.bits_per_dim)

    def encode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        z_q: (B, d, H, W) valeurs entières centrées (issues de FSQ), dans
             [-(L_i-1)/2, +(L_i-1)/2].
        Retourne b: (B, total_bits, H, W) flux binaire {0,1} (float, différentiable
        via l'identité pour permettre le passage du gradient jusqu'au canal).
        """
        b_list = []
        for i, (L_i, nbits) in enumerate(zip(self.levels, self.bits_per_dim)):
            zi = z_q[:, i:i + 1, :, :]
            offset = self._int_offset[i]
            range_max = self._range_size[i] - 1
            # décalage vers un index positif [0, range_max] par un OFFSET
            # ENTIER (pas la borne fractionnaire) -- affine, gradient=1
            idx_raw = zi + offset
            idx_rounded = idx_raw.round().clamp(0, range_max)

            # STE au niveau de l'index : forward = valeur entière arrondie,
            # backward = identité depuis idx_raw (donc depuis zi)
            idx_ste = idx_raw + (idx_rounded - idx_raw).detach()

            idx_int = idx_rounded.long()
            bits_hard = self._int_to_bits(idx_int, nbits).float()  # (B, nbits, H, W), pas de gradient

            # Proxy continu différentiable, diffusé sur les nbits canaux :
            # normalisé dans [0,1] pour rester à une échelle raisonnable.
            # (utilise range_max, l'offset ENTIER, pas la borne fractionnaire
            # (L_i-1) -- cohérent avec le décalage entier utilisé ci-dessus)
            denom = max(range_max, 1)
            idx_proxy = (idx_ste / denom).expand(-1, nbits, -1, -1)

            # STE global : forward = bits_hard (valeurs binaires correctes),
            # backward = gradient de idx_proxy (donc de zi), reparti sur les
            # nbits bits de cette dimension.
            bits_float = bits_hard + (idx_proxy - idx_proxy.detach())
            b_list.append(bits_float)
        return torch.cat(b_list, dim=1)

    def decode(self, b_hat: torch.Tensor) -> torch.Tensor:
        """
        b_hat: (B, total_bits, H, W) bits reçus (potentiellement corrompus).
        Retourne z_hat: (B, d, H, W) valeurs démappées, PAS ENCORE écrêtées
        (l'écrêtage strict est fait séparément dans latent_clamp.py).
        """
        z_list = []
        offset_bits = 0
        for i, (L_i, nbits) in enumerate(zip(self.levels, self.bits_per_dim)):
            bits = b_hat[:, offset_bits:offset_bits + nbits, :, :]
            idx = self._bits_to_int(bits, nbits)  # (B, 1, H, W) float
            zi = idx - self._int_offset[i]         # offset ENTIER, cohérent avec encode()
            z_list.append(zi)
            offset_bits += nbits
        return torch.cat(z_list, dim=1)

    @staticmethod
    @staticmethod
    def _int_to_bits(idx: torch.Tensor, nbits: int) -> torch.Tensor:
        """
        idx: (B, 1, H, W) long -> (B, nbits, H, W) {0,1}.
        Implémenté via division entière + modulo (`//`, `%`) plutôt que
        décalage binaire + ET bit-à-bit (`>>`, `&`) -- mathématiquement
        équivalent, mais l'export ONNX échoue explicitement sur
        `aten::__rshift__`/`aten::__and__`. `//`/`%` -> Div/Mod ONNX
        standards, exportables (voir net/emitter.py).
        """
        bits_list = []
        for k in range(nbits - 1, -1, -1):
            bit_k = torch.div(idx, 2 ** k, rounding_mode='floor') % 2
            bits_list.append(bit_k)
        return torch.cat(bits_list, dim=1)

    @staticmethod
    def _bits_to_int(bits: torch.Tensor, nbits: int) -> torch.Tensor:
        """
        bits: (B, nbits, H, W) valeurs dans [0,1] (0/1 dur si post-canal,
              continu si pré-canal via le STE de `encode`).
        Retourne idx: (B, 1, H, W) float.

        STE : forward = seuillage dur à 0.5 (comportement physique correct
        du démappeur), backward = identité depuis `bits` (laisse passer le
        gradient accumulé en amont -- crucial en Phase 1/2B où le canal est
        désactivé ou traité en perturbation douce, sinon l'encodeur ne
        recevrait jamais de gradient à travers decode()).
        """
        bits_hard = (bits > 0.5).float()
        bits_ste = bits + (bits_hard - bits).detach()  # forward=bits_hard, backward=identité
        weights = torch.tensor([2 ** k for k in range(nbits - 1, -1, -1)],
                                device=bits.device, dtype=torch.float32).view(1, -1, 1, 1)
        idx = (bits_ste * weights).sum(dim=1, keepdim=True)
        return idx
