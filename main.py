"""
Point d'entrée principal pour l'entraînement (AMP) du pipeline UWOC-SemCom.
Supporte le chargement dynamique des fichiers YAML (args.config) et
l'échantillonnage stratifié (StratifiedSampler) garantissant >= 50% de masques
par batch pour stabiliser le gradient Dice en phase 4.
"""
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from args import get_args
from config import Config
from net.pipeline import UWOCSemComPipeline
from datasets.coral_dataset import CoralDataset, DummyCoralDataset
from datasets.sampler import StratifiedSampler
from engine.amp_scheduler import AMPScheduler
from engine.phase1_fe import train_phase1_epoch
from engine.phase2_re import train_phase2a_epoch
from engine.phase3_tta import train_phase3_epoch
from engine.phase4_task import train_phase4_epoch


def apply_yaml_config(cfg: Config, yaml_path: str) -> Config:
    """
    Parse le fichier YAML de configuration et écrase les attributs correspondants de cfg,
    notamment cfg.levels (ou fsq_levels) et dimension latente cfg.d.
    Lie également dynamiquement B_e = R_b / 2 si R_b est modifié seul dans le YAML.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Fichier de configuration YAML introuvable : {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    if not params:
        return cfg

    for k, v in params.items():
        if k in ('fsq_levels', 'levels'):
            cfg.levels = list(v)
            cfg.d = len(cfg.levels)
        elif k == 'bottleneck_dim':
            cfg.d = int(v)
        else:
            setattr(cfg, k, v)

    # Assurer la cohérence d = len(levels)
    if hasattr(cfg, 'levels'):
        cfg.d = len(cfg.levels)

    # Si R_b est surchargé sans B_e, lier dynamiquement B_e = R_b / 2
    if 'R_b' in params and 'B_e' not in params:
        cfg.B_e = float(cfg.R_b) / 2.0

    return cfg


def evaluate_model(model, dataloader, device, link_distance):
    """Boucle d'évaluation standard (calcul des métriques pour l'article)."""
    from losses.reconstruction import psnr, ssim
    from utils.metrics import compute_miou, compute_top1_accuracy

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_ber = 0.0
    total_miou = 0.0
    total_acc = 0.0
    n_batches = len(dataloader)
    n_miou_batches = 0

    print(f"\n--- Évaluation du modèle à distance {link_distance} m ---")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            has_mask = batch['has_mask'].to(device).bool()

            outputs = model(images, link_distance=link_distance, phase='phase4', run_heads=True)

            x_hat = outputs['x_hat']
            total_psnr += psnr(images, x_hat).item()
            total_ssim += ssim(images, x_hat).item()
            total_ber += outputs['ber'].item()

            if has_mask.sum() > 0:
                miou = compute_miou(outputs['seg_logits'][has_mask], masks[has_mask], model.cfg.num_classes_seg)
                total_miou += miou
                n_miou_batches += 1

            acc = compute_top1_accuracy(outputs['cls_logits'], labels)
            total_acc += acc

    avg_psnr = total_psnr / max(1, n_batches)
    avg_ssim = total_ssim / max(1, n_batches)
    avg_ber = total_ber / max(1, n_batches)
    avg_miou = total_miou / max(1, n_miou_batches)
    avg_acc = total_acc / max(1, n_batches)

    print(f"Résultats d'évaluation :")
    print(f"  • PSNR moyen       : {avg_psnr:.2f} dB")
    print(f"  • SSIM moyen       : {avg_ssim:.4f}")
    print(f"  • BER moyen (p)    : {avg_ber:.4e}")
    print(f"  • mIoU Coraux      : {avg_miou:.4f}")
    print(f"  • Précision Top-1  : {avg_acc * 100:.2f} %")

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'ber': avg_ber,
        'miou': avg_miou,
        'accuracy': avg_acc,
    }


def main():
    args = get_args()
    cfg = Config()

    # 1. Écrasement des attributs de cfg par le fichier YAML passé en CLI via args.config
    if getattr(args, 'config', None):
        cfg = apply_yaml_config(cfg, args.config)
        print(f"Configuration YAML chargée depuis : {args.config}")
        print(f"Niveaux FSQ configurés : {cfg.levels} (dimension latente d={cfg.d})")

    # 2. Surcharges depuis la ligne de commande (si spécifiées)
    if getattr(args, 'data_root', None) is not None:
        cfg.data_root = args.data_root
    if getattr(args, 'channel_type', None) is not None:
        cfg.channel_type = args.channel_type
    if getattr(args, 'link_distance', None) is not None:
        cfg.link_distance = args.link_distance
    if getattr(args, 'batch_size', None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, 'lr', None) is not None:
        cfg.lr = args.lr
    if getattr(args, 'use_real_task_heads', None) is not None:
        cfg.use_real_task_heads = args.use_real_task_heads
    if getattr(args, 'seed', None) is not None:
        import random
        import numpy as np
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    device = torch.device(cfg.device)
    print(f"--- Démarrage UWOC-SemCom sur {device} (Mode : {args.mode.upper()}) ---")
    print(f"Distance du lien : {cfg.link_distance} m | Type de canal : {cfg.channel_type}")

    batch_size = getattr(cfg, 'batch_size', 4)
    lr = getattr(cfg, 'lr', 1e-4)

    # 3. Initialisation du Dataset
    train_split_dir = os.path.join(cfg.data_root, 'train') if cfg.data_root else None
    if train_split_dir and os.path.isdir(train_split_dir):
        print(f"Dataset réel trouvé : {cfg.data_root} (split='train')")
        dataset = CoralDataset(
            root=cfg.data_root, split='train', image_size=cfg.image_size,
            backbone_name=cfg.backbone_name,
        )
    else:
        print(f"AVERTISSEMENT : data_root introuvable ({cfg.data_root}) "
              f"-- repli sur DummyCoralDataset (données synthétiques).")
        dataset = DummyCoralDataset(
            num_samples=64, image_size=cfg.image_size,
            num_classes_seg=cfg.num_classes_seg, num_classes_cls=cfg.num_classes_cls,
            mask_ratio=0.5,
        )

    # 4. Échantillonneur stratifié (Stratified Sampler)
    sampler = StratifiedSampler(
        dataset=dataset,
        batch_size=batch_size,
        min_pos_ratio=0.5,
        shuffle=(args.mode == 'train'),
        seed=getattr(args, 'seed', None),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 5. Instanciation du Modèle complet
    model = UWOCSemComPipeline(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reprise depuis un checkpoint (--resume)
    from utils.checkpoint import save_checkpoint, load_checkpoint
    if getattr(args, 'resume', None) and os.path.isfile(args.resume):
        load_checkpoint(model, args.resume, optimizer=optimizer if args.mode == 'train' else None, map_location=device)
        print(f"Checkpoint chargé avec succès depuis : {args.resume}")

    # Mode Évaluation pure
    if args.mode == 'eval':
        evaluate_model(model, dataloader, device, cfg.link_distance)
        return

    # 6. Planificateur AMP et boucle d'entraînement
    total_epochs = args.epochs if getattr(args, 'epochs', None) is not None else 10
    scheduler = AMPScheduler(total_epochs=total_epochs)
    ckpt_dir = os.path.join(os.getcwd(), 'checkpoints')

    if getattr(args, 'phase', None) is not None:
        phase_num = args.phase
        print(f"\nExécution de la Phase ciblée {phase_num} sur {total_epochs} époque(s)...")
        for epoch in range(total_epochs):
            if phase_num == 1:
                loss = train_phase1_epoch(model, dataloader, optimizer, device)
            elif phase_num == 2:
                loss = train_phase2a_epoch(model, dataloader, optimizer, device, cfg.link_distance, beta=1.0)
            elif phase_num == 3:
                loss = train_phase3_epoch(model, dataloader, optimizer, device, cfg.link_distance, beta=1.0)
            elif phase_num == 4:
                loss = train_phase4_epoch(model, dataloader, optimizer, device, cfg.link_distance)
            print(f"[Phase {phase_num} - Epoch {epoch+1}/{total_epochs}] Loss : {loss:.4f}")
            save_checkpoint(model, optimizer, epoch + 1, f'phase{phase_num}', ckpt_dir, tag='last')
    else:
        for epoch in range(total_epochs):
            phase_id, phase_name = scheduler.get_phase(epoch)
            print(f"\n[Epoch {epoch+1}/{total_epochs}] Phase active : {phase_id.upper()} - {phase_name}")

            if phase_id == 'phase1':
                loss = train_phase1_epoch(model, dataloader, optimizer, device)
            elif phase_id == 'phase2a':
                loss = train_phase2a_epoch(model, dataloader, optimizer, device, cfg.link_distance, beta=1.0)
            elif phase_id == 'phase3':
                loss = train_phase3_epoch(model, dataloader, optimizer, device, cfg.link_distance, beta=1.0)
            elif phase_id == 'phase4':
                loss = train_phase4_epoch(model, dataloader, optimizer, device, cfg.link_distance)

            print(f"--> Loss moyenne : {loss:.4f}")
            save_checkpoint(model, optimizer, epoch + 1, phase_id, ckpt_dir, tag='last')

    print(f"\nEntraînement terminé. Checkpoints sauvegardés dans : {ckpt_dir}")


if __name__ == "__main__":
    main()
