"""
Logique d'entraînement de la Phase 2 : Renforcement de Robustesse.
Étape 2A : Gel de l'encodeur, Entraînement IRSNet + G_psi sous bruit et attaque de masque.
"""
import torch
from losses.latent_align import compute_phase2a_loss


def train_phase2a_epoch(model, dataloader, optimizer, device, link_distance, beta=1.0):
    model.train()

    # 1. Gel de l'encodeur (Backbone + fusion ASFF + Edge Bottleneck)
    model.backbone.eval()
    for param in model.backbone.parameters():
        param.requires_grad = False
    if model.multiscale_fusion is not None:
        model.multiscale_fusion.eval()
        for param in model.multiscale_fusion.parameters():
            param.requires_grad = False
    model.edge_bottleneck.eval()
    for param in model.edge_bottleneck.parameters():
        param.requires_grad = False

    # 2. Activation des gradients pour IRSNet et Generator
    for param in model.irsnet.parameters():
        param.requires_grad = True
    for param in model.generator.parameters():
        param.requires_grad = True

    # 3. Gel des têtes de tâches aval en mode eval
    model.seg_head.eval()
    for param in model.seg_head.parameters():
        param.requires_grad = False
    model.cls_head.eval()
    for param in model.cls_head.parameters():
        param.requires_grad = False

    total_loss = 0.0

    for batch in dataloader:
        images = batch['image'].to(device)

        optimizer.zero_grad()

        # Forward pass (BFP + MATK actifs, sans calcul inutile des têtes de tâche)
        outputs = model(images, link_distance=link_distance, phase='phase2a', run_heads=False)

        loss = compute_phase2a_loss(
            x=images,
            x_hat=outputs['x_hat'],
            z_q=outputs['z_q'],
            z_restored=outputs['z_restored'],
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_phase2b_epoch(model, dataloader, optimizer, device, link_distance, beta=1.0):
    """
    Étape 2B (RE) : Mise à jour conjointe Encodeur + Décodeur sous bruit physique (BFP), sans MATK.
    Permet à l'émetteur d'adapter ses représentations latentes au canal optique sous-marin.
    """
    model.train()

    # Dégel complet de l'encodeur et du décodeur
    for param in model.parameters():
        param.requires_grad = True

    # Respect du gel éventuel du backbone
    backbone_frozen = getattr(getattr(model, 'cfg', None), 'backbone_freeze', False)
    if backbone_frozen:
        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Gel des têtes de tâches aval en mode eval
    model.seg_head.eval()
    for param in model.seg_head.parameters():
        param.requires_grad = False
    model.cls_head.eval()
    for param in model.cls_head.parameters():
        param.requires_grad = False

    total_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)

        optimizer.zero_grad()

        # phase2b : canal actif, sans masque MATK
        outputs = model(images, link_distance=link_distance, phase='phase2b', run_heads=False)

        loss = compute_phase2a_loss(
            x=images,
            x_hat=outputs['x_hat'],
            z_q=outputs['z_q'],
            z_restored=outputs['z_restored'],
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
