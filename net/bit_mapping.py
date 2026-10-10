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
        self.levels = [int(l) for l in levels]
        # Conforme à Mentzer et al. (App. A.1) : l'offset entier est L // 2,
        # ce qui mappe [-(L // 2), (L - 1) - (L // 2)] exactement sur [0, L - 1].
        self._int_offset = [l // 2 for l in self.levels]
        self.bits_per_dim = [max(1, math.ceil(math.log2(l))) for l in self.levels]
        self.total_bits = sum(self.bits_per_dim)

    def encode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        z_q: (B, d, H, W) valeurs entières centrées (issues de FSQ), dans
             [-(L_i // 2), (L_i - 1) - (L_i // 2)].
        Retourne b: (B, total_bits, H, W) flux binaire {0,1} (float, différentiable
        via STE pour permettre le passage du gradient jusqu'au canal).
        """
        b_list = []
        for i, (L_i, nbits) in enumerate(zip(self.levels, self.bits_per_dim)):
            zi = z_q[:, i:i + 1, :, :]
            offset = self._int_offset[i]
            range_max = L_i - 1
            # Décalage vers un index positif [0, L_i - 1]
            idx_raw = zi + offset
            idx_rounded = idx_raw.round().clamp(0, range_max)

            # STE au niveau de l'index : forward = valeur entière arrondie,
            # backward = identité depuis idx_raw (donc depuis zi)
            idx_ste = idx_raw + (idx_rounded - idx_raw).detach()

            idx_int = idx_rounded.long()
            bits_hard = self._int_to_bits(idx_int, nbits).float()  # (B, nbits, H, W)

            # Proxy continu différentiable dans [0, 1]
            denom = max(range_max, 1)
            idx_proxy = (idx_ste / denom).expand(-1, nbits, -1, -1)

            # STE global : forward = bits_hard, backward = gradient de idx_proxy
            bits_float = bits_hard + (idx_proxy - idx_proxy.detach())
            b_list.append(bits_float)
        return torch.cat(b_list, dim=1)

    def decode(self, b_hat: torch.Tensor) -> torch.Tensor:
        """
        b_hat: (B, total_bits, H, W) bits reçus (potentiellement corrompus).
        Retourne z_hat: (B, d, H, W) valeurs démappées.
        """
        z_list = []
        offset_bits = 0
        for i, (L_i, nbits) in enumerate(zip(self.levels, self.bits_per_dim)):
            bits = b_hat[:, offset_bits:offset_bits + nbits, :, :]
            idx = self._bits_to_int(bits, nbits)  # (B, 1, H, W) float
            # Sécurité contre les sauts de bits corrompus : borne à [0, L_i - 1]
            idx_clamped = idx.clamp(0, L_i - 1)
            zi = idx_clamped - self._int_offset[i]
            z_list.append(zi)
            offset_bits += nbits
        return torch.cat(z_list, dim=1)

    @staticmethod
    def _int_to_bits(idx: torch.Tensor, nbits: int) -> torch.Tensor:
        """
        idx: (B, 1, H, W) long -> (B, nbits, H, W) {0,1}.
        Implémenté via division entière + modulo (`//`, `%`) plutôt que
        décalage binaire + ET bit-à-bit (`>>`, `&`) pour compatibilité ONNX.
        """
        bits_list = []
        for k in range(nbits - 1, -1, -1):
            bit_k = torch.div(idx, 2 ** k, rounding_mode='floor') % 2
            bits_list.append(bit_k)
        return torch.cat(bits_list, dim=1)

    @staticmethod
    def _bits_to_int(bits: torch.Tensor, nbits: int) -> torch.Tensor:
        """
        bits: (B, nbits, H, W) valeurs dans [0,1].
        Retourne idx: (B, 1, H, W) float via STE.
        """
        bits_hard = (bits > 0.5).float()
        bits_ste = bits + (bits_hard - bits).detach()
        weights = torch.tensor([2 ** k for k in range(nbits - 1, -1, -1)],
                                device=bits.device, dtype=torch.float32).view(1, -1, 1, 1)
        idx = (bits_ste * weights).sum(dim=1, keepdim=True)
        return idx
