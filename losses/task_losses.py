"""
Pertes des têtes de tâches aval (Phase 4) : Cross-Entropy + Dice (segmentation)
+ Cross-Entropy (classification).
L_P4 = CE(seg) + lambda_dice * Dice(seg) + lambda_cls * CE(cls)
"""
import torch
import torch.nn.functional as F
from losses.latent_align import compute_phase4_loss


def dice_loss(logits, target, num_classes, eps=1.0):
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (probs * target_onehot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


def task_loss(seg_logits, seg_target, cls_logits, cls_target,
               num_classes_seg=None, lambda_dice=1.0, lambda_cls=0.5, has_mask=None):
    """Calcule la perte de tâche via compute_phase4_loss (perte unifiée du projet)."""
    total = compute_phase4_loss(
        seg_logits=seg_logits,
        seg_masks=seg_target,
        cls_logits=cls_logits,
        cls_labels=cls_target,
        has_mask=has_mask,
        lambda_dice=lambda_dice,
        lambda_cls=lambda_cls,
    )
    return total, {'total_task_loss': total.item()}
