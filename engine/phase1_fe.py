"""
Phase 1: Extraction de Caractéristiques (FE).
Entraînement conjoint Encodeur + Décodeur sans bruit.
Optimisation : Perte de reconstruction L1.
"""
import torch
from losses.latent_align import compute_phase1_loss


def train_phase1_epoch(model, dataloader, optimizer, device):
    model.train()

    # Activation des gradients pour encodeur et décodeur
    for param in model.parameters():
        param.requires_grad = True

    # On gèle les têtes de tâches aval pour le moment
    for param in model.seg_head.parameters():
        param.requires_grad = False
    for param in model.cls_head.parameters():
        param.requires_grad = False

    total_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)

        optimizer.zero_grad()
        # Phase 1 = régime analogique parfait (sans bruit de canal)
        outputs = model(images, link_distance=0, phase='phase1')

        loss = compute_phase1_loss(x=images, x_hat=outputs['x_hat'])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
