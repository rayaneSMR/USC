"""
Métriques d'évaluation sémantique et de tâche : mIoU, mPA (CoralMask),
Top-1 accuracy, F1.
"""
import torch


def compute_miou(pred_logits, target, num_classes, eps=1e-6):
    pred = pred_logits.argmax(dim=1)
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            continue
        ious.append((intersection / (union + eps)).item())
    return sum(ious) / len(ious) if ious else 0.0


def compute_mpa(pred_logits, target, num_classes, eps=1e-6):
    pred = pred_logits.argmax(dim=1)
    accs = []
    for c in range(num_classes):
        target_c = (target == c)
        if target_c.sum() == 0:
            continue
        correct = ((pred == c) & target_c).sum().float()
        accs.append((correct / (target_c.sum().float() + eps)).item())
    return sum(accs) / len(accs) if accs else 0.0


def compute_top1_accuracy(cls_logits, target):
    pred = cls_logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def compute_f1(cls_logits, target, num_classes, eps=1e-6):
    pred = cls_logits.argmax(dim=1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().float()
        fp = ((pred == c) & (target != c)).sum().float()
        fn = ((pred != c) & (target == c)).sum().float()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1.item())
    return sum(f1s) / len(f1s) if f1s else 0.0
