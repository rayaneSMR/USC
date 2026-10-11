"""
Phase 4 : Adaptation des Têtes de Tâches Aval.
Gel de la chaîne codec et du canal.
Entraînement des têtes de segmentation ET de classification sur x_hat.
Optimisation : Cross-Entropy + Dice Loss (seg) + Cross-Entropy (cls).
"""
import torch
from losses.latent_align import compute_phase4_loss


def train_phase4_epoch(model, dataloader, optimizer, device, link_distance):
    # 1. Gel COMPLET de la chaîne de transmission et de reconstruction en mode eval
    # (empêche BatchNorm et Dropout de modifier les statistiques de l'encodeur/décodeur)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 2. Activation uniquement des têtes de tâches aval
    model.seg_head.train()
    for param in model.seg_head.parameters():
        param.requires_grad = True
    model.cls_head.train()
    for param in model.cls_head.parameters():
        param.requires_grad = True

    # Si l'encodeur de la tête est gelé (ex. backbone MiT de SegFormerHead)
    freeze_encoders = getattr(getattr(model, 'cfg', None), 'freeze_task_head_encoders', True)
    if freeze_encoders:
        for head in (model.seg_head, model.cls_head):
            if hasattr(head, 'model') and hasattr(head.model, 'segformer'):
                head.model.segformer.eval()
                for p in head.model.segformer.parameters():
                    p.requires_grad = False
            elif hasattr(head, 'encoder') and head.encoder is not None:
                head.encoder.eval()
                for p in head.encoder.parameters():
                    p.requires_grad = False

    total_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device)
        has_mask = batch['has_mask'].to(device)

        optimizer.zero_grad()

        # Passage dans le modèle complet avec bruit de canal (pas de MATK en phase4)
        outputs = model(images, link_distance=link_distance, phase='phase4', run_heads=True)

        loss = compute_phase4_loss(
            seg_logits=outputs['seg_logits'], seg_masks=masks,
            cls_logits=outputs['cls_logits'], cls_labels=labels,
            has_mask=has_mask,
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
