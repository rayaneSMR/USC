"""
Point d'entrée principal pour l'entraînement (AMP) du pipeline UWOC-SemCom.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from net.pipeline import UWOCSemComPipeline
from datasets.coral_dataset import CoralDataset, DummyCoralDataset
from engine.amp_scheduler import AMPScheduler
from engine.phase1_fe import train_phase1_epoch
from engine.phase2_re import train_phase2a_epoch
from engine.phase3_tta import train_phase3_epoch
from engine.phase4_task import train_phase4_epoch

def main():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"--- Démarrage de l'entraînement UWOC-SemCom sur {device} ---")

    # 1. Initialisation du Dataset -- vrai CoralDataset si data_root est
    # défini et existe (sortie de coral_dataset_merge_pipeline_v2.ipynb),
    # sinon repli automatique sur DummyCoralDataset (données synthétiques,
    # pour tester le pipeline sans dataset réel).
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
        )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Instanciation du Modèle complet
    model = UWOCSemComPipeline(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. Planificateur AMP (Alternating Multi-Phase)
    total_epochs = 10
    scheduler = AMPScheduler(total_epochs=total_epochs)
    
    # 4. Boucle d'entraînement
    for epoch in range(total_epochs):
        phase_id, phase_name = scheduler.get_phase(epoch)
        print(f"\n[Epoch {epoch+1}/{total_epochs}] Phase active : {phase_id.upper()} - {phase_name}")
        
        if phase_id == 'phase1':
            loss = train_phase1_epoch(model, dataloader, optimizer, device)
        elif phase_id == 'phase2a':
            # beta = 1.0 par défaut pour l'alignement latent
            loss = train_phase2a_epoch(model, dataloader, optimizer, device, cfg.link_distance, beta=1.0)
        elif phase_id == 'phase3':
            loss = train_phase3_epoch(model, dataloader, optimizer, device, cfg.link_distance, beta=1.0)
        elif phase_id == 'phase4':
            loss = train_phase4_epoch(model, dataloader, optimizer, device, cfg.link_distance)
            
        print(f"--> Loss moyenne : {loss:.4f}")

if __name__ == "__main__":
    main()
