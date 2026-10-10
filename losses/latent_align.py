"""
Fonctions de perte pour l'entraînement multi-phases (AMP).
"""
import torch
import torch.nn.functional as F

def compute_phase1_loss(x, x_hat):
    """
    Phase 1 (FE) : Perte de reconstruction de base L1 en régime parfait.
    """
    return F.l1_loss(x_hat, x)

def compute_phase2a_loss(x, x_hat, z_q, z_restored, beta=1.0):
    """
    Phase 2A (RE) : Combinaison de la fidélité visuelle et de l'alignement latent.
    x_hat : image reconstruite
    x : image originale
    z_q : espace latent original (sans bruit)
    z_restored : espace latent restauré par l'IRSNet
    """
    loss_recon = F.l1_loss(x_hat, x)
    loss_latent = F.l1_loss(z_restored, z_q.detach())
    return loss_recon + beta * loss_latent

def compute_phase4_loss(seg_logits, seg_masks, cls_logits=None, cls_labels=None,
                         has_mask=None, lambda_dice=1.0, lambda_cls=0.5):
    """
    Phase 4 : Tâches spécifiques -- L_P4 = CE(seg) + lambda_dice*Dice(seg)
    + lambda_cls*CE(cls) (§4). cls_logits/cls_labels optionnels pour
    compatibilité arrière (pipelines sans tête de classification).

    has_mask: (B,) bool tensor, optionnel. Le dataset fusionné (voir
    datasets/coral_dataset.py) mélange des images avec vrai masque de
    segmentation (sources COCO) et des images sans masque (source
    classification pure) -- ces dernières ont un masque de ZÉROS
    placeholder qui NE DOIT PAS être utilisé comme cible de segmentation
    (il ne signifie pas "tout est fond"). Si `has_mask` est fourni, la
    perte de segmentation (CE + Dice) est calculée par échantillon puis
    moyennée UNIQUEMENT sur les échantillons où has_mask=True -- le batch
    peut contenir un mélange des deux types, seule la partie classification
    reste supervisée sur les échantillons sans masque. Si `has_mask` est
    None (comportement par défaut, compatibilité arrière), tous les
    échantillons du batch sont supposés avoir un masque réel.
    """
    if has_mask is None:
        has_mask = torch.ones(seg_masks.shape[0], dtype=torch.bool, device=seg_masks.device)
    else:
        has_mask = has_mask.bool().to(seg_masks.device)

    n_valid = has_mask.sum()

    if n_valid > 0:
        # CE par échantillon (reduction='none'), moyenné spatialement
        # (H,W) puis filtré sur has_mask avant de moyenner sur le batch --
        # sans ce filtrage, un échantillon has_mask=0 (masque de zéros
        # placeholder) fausserait la perte en apprenant "tout est fond".
        ce_per_sample = F.cross_entropy(seg_logits, seg_masks, reduction='none').mean(dim=(1, 2))
        ce_loss = ce_per_sample[has_mask].mean()

        probs = F.softmax(seg_logits, dim=1)
        targets_one_hot = F.one_hot(seg_masks, num_classes=seg_logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        smooth = 1.0
        dice_per_sample = 1.0 - ((2. * intersection + smooth) / (union + smooth)).mean(dim=1)
        dice_loss = dice_per_sample[has_mask].mean()

        total = ce_loss + lambda_dice * dice_loss
    else:
        # Aucun échantillon du batch n'a de masque réel (garde-fou si le StratifiedSampler n'est pas utilisé)
        total = torch.zeros((), device=seg_logits.device, requires_grad=True)

    if cls_logits is not None and cls_labels is not None:
        # La classification, elle, est toujours supervisée sur TOUT le
        # batch -- 100% des images du dataset fusionné ont un label de
        # classification, qu'un masque existe ou non.
        ce_cls = F.cross_entropy(cls_logits, cls_labels)
        total = total + lambda_cls * ce_cls

    return total
