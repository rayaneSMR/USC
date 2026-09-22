"""
IRSNet — réseau de restauration latente (§3.3), U-Net résiduel multi-échelle.
Prend le tenseur latent écrêté (bruité par le canal) et élimine les motifs de
distorsion introduits par les sauts de bits, reconstruisant la cohérence
spatiale avant décodage visuel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, skip):
        x = self.act(self.norm(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            # Résolution d'entrée non divisible par 2^depth (point 6) :
            # un Down (stride=2, arrondi par troncature) suivi d'un Up
            # (stride=2 exact) ne redonne pas exactement la taille de la
            # skip connection correspondante pour une taille impaire/non
            # multiple de 2^depth. On recale x sur la taille de skip par
            # interpolation plutôt que de laisser l'addition planter.
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return x + skip


class IRSNet(nn.Module):
    """
    U-Net résiduel léger, opère directement sur les cartes latentes de faible
    dimension (d ≤ 8 canaux), pas des images RGB -> peu de downsampling.
    """

    def __init__(self, in_channels, base_channels=32, depth=2, num_resblocks=2):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.res_enc = nn.ModuleList()
        ch = base_channels
        for _ in range(depth):
            self.res_enc.append(nn.Sequential(*[ResBlock(ch) for _ in range(num_resblocks)]))
            self.downs.append(Down(ch, ch * 2))
            ch *= 2

        self.bottleneck = nn.Sequential(*[ResBlock(ch) for _ in range(num_resblocks)])

        self.ups = nn.ModuleList()
        self.res_dec = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(ch, ch // 2))
            ch //= 2
            self.res_dec.append(nn.Sequential(*[ResBlock(ch) for _ in range(num_resblocks)]))

        self.head = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, z_clamped):
        x = self.stem(z_clamped)
        skips = []
        for res, down in zip(self.res_enc, self.downs):
            x = res(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, res, skip in zip(self.ups, self.res_dec, reversed(skips)):
            x = up(x, skip)
            x = res(x)

        z_restored = self.head(x)
        return z_restored
