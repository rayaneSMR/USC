"""
Pertes des têtes de tâches aval (Phase 4) : Cross-Entropy + Dice (segmentation)
+ Cross-Entropy (classification).
L_P4 = CE(seg) + lambda_dice * Dice(seg) + lambda_cls * CE(cls)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (probs * target_onehot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def task_loss(seg_logits, seg_target, cls_logits, cls_target,
               num_classes_seg, lambda_dice=1.0, lambda_cls=0.5):
    ce_seg = F.cross_entropy(seg_logits, seg_target)
    dice = dice_loss(seg_logits, seg_target, num_classes_seg)
    ce_cls = F.cross_entropy(cls_logits, cls_target)

    total = ce_seg + lambda_dice * dice + lambda_cls * ce_cls
    return total, {
        'ce_seg': ce_seg.item(),
        'dice': dice.item(),
        'ce_cls': ce_cls.item(),
    }
