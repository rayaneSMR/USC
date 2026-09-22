"""
Sauvegarde / chargement de checkpoints par phase.
"""
import os
import torch


def save_checkpoint(model, optimizer, epoch, phase, ckpt_dir, tag='last'):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'phase{phase}_{tag}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
        'phase': phase,
    }, path)
    return path


def load_checkpoint(model, path, optimizer=None, map_location='cpu', strict=True):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'], strict=strict)
    if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('phase', None)
