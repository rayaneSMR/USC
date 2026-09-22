"""
Phase 3 : Alignement Entraînement-Test (TTA).
Désactivation de MATK. Ajustement fin de l'IRSNet et du décodeur sous BFP.
"""
import torch
from losses.latent_align import compute_phase2a_loss


def train_phase3_epoch(model, dataloader, optimizer, device, link_distance, beta=1.0):
    model.train()

    # Comme en phase 2A, on gèle l'encodeur (dont la fusion ASFF)
    for param in model.backbone.parameters():
        param.requires_grad = False
    if model.multiscale_fusion is not None:
        for param in model.multiscale_fusion.parameters():
            param.requires_grad = False
    for param in model.edge_bottleneck.parameters():
        param.requires_grad = False

    # On active IRSNet et Generator
    for param in model.irsnet.parameters():
        param.requires_grad = True
    for param in model.generator.parameters():
        param.requires_grad = True

    total_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)

        optimizer.zero_grad()

        # phase='phase3' désactive MATK dans le pipeline
        outputs = model(images, link_distance=link_distance, phase='phase3')

        # Même perte qu'en Phase 2A (L1 visuel + L1 latent) mais sur des données sans masque artificiel
        loss = compute_phase2a_loss(
            x=images, x_hat=outputs['x_hat'],
            z_q=outputs['z_q'], z_restored=outputs['z_restored'], beta=beta,
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
