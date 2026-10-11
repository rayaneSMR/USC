"""
Tests unitaires pour le protocole d'entraînement multi-phases (AMP, TTA, MATK, Task Adaptation)
selon arXiv:2408.04972 et les spécifications du projet.
"""
import torch
from config import Config
from net.pipeline import UWOCSemComPipeline
from net.mask_attack import MATK
from engine.amp_scheduler import AMPScheduler
from engine.phase1_fe import train_phase1_epoch
from engine.phase2_re import train_phase2a_epoch
from engine.phase3_tta import train_phase3_epoch
from engine.phase4_task import train_phase4_epoch
from losses.latent_align import compute_phase4_loss


def test_matk_eval_mode():
    matk = MATK(ratio_factor=2.0)
    z = torch.ones(2, 5, 8, 8)
    p = torch.tensor(0.2)

    # En mode train, MATK applique le masquage
    matk.train()
    z_masked = matk(z, p)
    assert not torch.all(z_masked == z), "En mode train, MATK doit masquer des tokens"

    # En mode eval, MATK ne doit rien masquer
    matk.eval()
    z_eval = matk(z, p)
    assert torch.all(z_eval == z), "En mode eval, MATK ne doit JAMAIS masquer de tokens"
    print("test_matk_eval_mode OK")


def test_pipeline_run_heads_flag():
    cfg = Config()
    cfg.backbone_name = 'simple_cnn'
    cfg.device = 'cpu'
    cfg.image_size = 32
    cfg.use_multiscale_fusion = False
    cfg.use_real_task_heads = False

    pipeline = UWOCSemComPipeline(cfg)
    x = torch.randn(1, 3, 32, 32)

    # run_heads=False (Phases 1, 2, 3)
    out_no_heads = pipeline(x, link_distance=0, phase='phase1', run_heads=False)
    assert out_no_heads['seg_logits'] is None
    assert out_no_heads['cls_logits'] is None

    # run_heads=True (Phase 4 / inférence)
    out_with_heads = pipeline(x, link_distance=0, phase='phase4', run_heads=True)
    assert out_with_heads['seg_logits'] is not None
    assert out_with_heads['cls_logits'] is not None
    print("test_pipeline_run_heads_flag OK")


def test_amp_scheduler_no_skipped_phases():
    # Pour total_epochs = 4
    sched4 = AMPScheduler(total_epochs=4)
    phases4 = [sched4.get_phase(ep)[0] for ep in range(4)]
    assert phases4 == ['phase1', 'phase2a', 'phase3', 'phase4'], f"Attendu 1 par phase, obtenu {phases4}"

    # Pour total_epochs = 10
    sched10 = AMPScheduler(total_epochs=10)
    phases10 = [sched10.get_phase(ep)[0] for ep in range(10)]
    assert 'phase1' in phases10
    assert 'phase2a' in phases10
    assert 'phase3' in phases10
    assert 'phase4' in phases10
    print("test_amp_scheduler_no_skipped_phases OK")


def test_phase_freeze_modes():
    cfg = Config()
    cfg.backbone_name = 'simple_cnn'
    cfg.device = 'cpu'
    cfg.image_size = 32
    cfg.use_multiscale_fusion = False
    cfg.use_real_task_heads = False

    pipeline = UWOCSemComPipeline(cfg)
    dummy_loader = [{'image': torch.randn(2, 3, 32, 32),
                     'mask': torch.zeros(2, 32, 32, dtype=torch.long),
                     'label': torch.zeros(2, dtype=torch.long),
                     'has_mask': torch.tensor([1, 1], dtype=torch.int64)}]
    optimizer = torch.optim.SGD(pipeline.parameters(), lr=1e-3)

    # Test Phase 3 : Backbone en eval() et requires_grad=False
    train_phase3_epoch(pipeline, dummy_loader, optimizer, 'cpu', link_distance=0)
    assert not pipeline.backbone.training, "Backbone doit être en eval() pendant Phase 3"
    assert not pipeline.edge_bottleneck.training, "EdgeBottleneck doit être en eval() pendant Phase 3"
    assert pipeline.irsnet.training, "IRSNet doit être en train() pendant Phase 3"
    assert pipeline.generator.training, "Generator doit être en train() pendant Phase 3"

    # Test Phase 4 : Codec complet en eval(), têtes de tâches en train()
    train_phase4_epoch(pipeline, dummy_loader, optimizer, 'cpu', link_distance=0)
    assert not pipeline.backbone.training, "Backbone doit être en eval() pendant Phase 4"
    assert not pipeline.edge_bottleneck.training, "EdgeBottleneck doit être en eval() pendant Phase 4"
    assert not pipeline.irsnet.training, "IRSNet doit être en eval() pendant Phase 4"
    assert not pipeline.generator.training, "Generator doit être en eval() pendant Phase 4"
    assert pipeline.seg_head.training, "seg_head doit être en train() pendant Phase 4"
    assert pipeline.cls_head.training, "cls_head doit être en train() pendant Phase 4"
    print("test_phase_freeze_modes OK")


def test_phase4_loss_with_integer_has_mask():
    seg_logits = torch.randn(4, 3, 16, 16, requires_grad=True)
    seg_masks = torch.randint(0, 3, (4, 16, 16))
    cls_logits = torch.randn(4, 3, requires_grad=True)
    cls_labels = torch.randint(0, 3, (4,))
    # Integer tensor comme issu d'un DataLoader typique
    has_mask = torch.tensor([1, 1, 0, 0], dtype=torch.int64)

    loss = compute_phase4_loss(seg_logits, seg_masks, cls_logits, cls_labels, has_mask=has_mask)
    loss.backward()

    assert seg_logits.grad[2:].abs().sum().item() == 0.0, "Gradient doit être nul sur has_mask=0 même avec int tensor"
    assert seg_logits.grad[:2].abs().sum().item() > 0.0, "Gradient non-nul sur has_mask=1"
    print("test_phase4_loss_with_integer_has_mask OK")


def test_phase4_segformer_encoder_freeze():
    """Vérifie que l'encodeur MiT de SegFormer est bien gelé en Phase 4 quand freeze_task_head_encoders=True."""
    from net.heads import SegFormerSegmentationHead, SegFormerClassificationHead
    cfg = Config()
    cfg.backbone_name = 'simple_cnn'
    cfg.image_size = 32
    cfg.use_multiscale_fusion = False
    cfg.use_real_task_heads = True
    cfg.freeze_task_head_encoders = True
    cfg.device = 'cpu'

    pipeline = UWOCSemComPipeline(cfg)
    dummy_loader = [{'image': torch.randn(2, 3, 32, 32),
                     'mask': torch.zeros(2, 32, 32, dtype=torch.long),
                     'label': torch.zeros(2, dtype=torch.long),
                     'has_mask': torch.tensor([1, 1], dtype=torch.int64)}]
    optimizer = torch.optim.SGD(pipeline.parameters(), lr=1e-3)

    train_phase4_epoch(pipeline, dummy_loader, optimizer, 'cpu', link_distance=0)
    assert not next(pipeline.seg_head.model.segformer.parameters()).requires_grad
    assert not next(pipeline.cls_head.model.segformer.parameters()).requires_grad
    print("test_phase4_segformer_encoder_freeze OK")


if __name__ == '__main__':
    test_matk_eval_mode()
    test_pipeline_run_heads_flag()
    test_amp_scheduler_no_skipped_phases()
    test_phase_freeze_modes()
    test_phase4_loss_with_integer_has_mask()
    test_phase4_segformer_encoder_freeze()
