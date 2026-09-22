"""
Trace le BER théorique p en fonction de la distance de liaison ou du SNR
optique, en utilisant Channel.optimal_threshold_ber (§3.2).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_ber_vs_distance(channel, distances, shape=(1, 8, 32, 32), device='cpu'):
    """
    channel: instance de net.channel.Channel
    distances: liste/array de distances en mètres
    Retourne (distances, ber_values)
    """
    bers = []
    bits = torch.randint(0, 2, shape, dtype=torch.float32, device=device)
    for L in distances:
        with torch.no_grad():
            _, p = channel(bits, link_distance=L)
        bers.append(p.item())
    return np.array(distances), np.array(bers)


def plot_ber_curve(distances, bers, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.semilogy(distances, bers, marker='o')
    plt.xlabel('Distance de liaison (m)')
    plt.ylabel('BER (p)')
    plt.title('BER vs distance — canal UWOC IM/DD')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
