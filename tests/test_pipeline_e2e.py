"""
Tests end-to-end : x -> pipeline complet -> x_hat, seg_logits, cls_logits,
avec formes cohérentes, gradient, ASFF branché, vraies têtes, 4 phases.
"""
import torch
from config import Config
from net.pipeline import UWOCSemComPipeline


def test_pipeline_forward_lightweight():
    """Combinaison légère (simple_cnn + stubs) : test de fumée rapide,
    vérifie l'assemblage/les formes, pas la qualité des résultats."""
    cfg = Config()
    cfg.backbone_name = 'simple_cnn'
    cfg.device = 'cpu'
    cfg.image_size = 128
    cfg.use_multiscale_fusion = False
    cfg.use_real_task_heads = False

    pipeline = UWOCSemComPipeline(cfg)
    x = torch.randn(1, 3, 128, 128)

    out = pipeline(x, link_distance=10.0, phase='phase2a')

    assert out['x_hat'].shape == (1, 3, 128, 128)
    assert out['seg_logits'].shape[1] == cfg.num_classes_seg
    assert out['cls_logits'].shape == (1, cfg.num_classes_cls)
    print('test_pipeline_forward_lightweight OK, x_hat shape =', out['x_hat'].shape)


def test_pipeline_forward_yolo26_asff_gradient():
    """Backbone réel YOLO26 + fusion ASFF branchée : vérifie le gradient
    bout-en-bout à travers tout l'assemblage (pas juste les formes)."""
    cfg = Config()
    cfg.backbone_name = 'yolo26'
    cfg.device = 'cpu'
    cfg.image_size = 128
    cfg.use_multiscale_fusion = True
    cfg.use_real_task_heads = False

    pipeline = UWOCSemComPipeline(cfg)
    assert pipeline.use_multiscale_fusion, "ASFF devrait être actif avec YOLO26"

    x = torch.randn(1, 3, 128, 128, requires_grad=True)
    out = pipeline(x, link_distance=10.0, phase='phase2a')
    loss = out['x_hat'].sum() + out['seg_logits'].sum() + out['cls_logits'].sum()
    loss.backward()

    assert x.grad is not None
    print('test_pipeline_forward_yolo26_asff_gradient OK')


def test_pipeline_forward_real_heads_small_image():
    """Vraies têtes SegFormer (§3.4) sur une petite image pour rester dans
    la RAM d'un container de dev."""
    cfg = Config()
    cfg.backbone_name = 'simple_cnn'
    cfg.device = 'cpu'
    cfg.image_size = 64
    cfg.use_multiscale_fusion = False
    cfg.use_real_task_heads = True
    cfg.num_classes_seg = 2
    cfg.num_classes_cls = 3

    pipeline = UWOCSemComPipeline(cfg)
    pipeline.eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        out = pipeline(x, link_distance=10.0, phase='phase4')

    assert out['seg_logits'].shape == (1, 2, 64, 64)
    assert out['cls_logits'].shape == (1, 3)
    print('test_pipeline_forward_real_heads_small_image OK')


def test_pipeline_phase1_bypasses_channel():
    """Phase 1 : pas de canal -> BER doit rester à 0."""
    cfg = Config()
    cfg.backbone_name = 'simple_cnn'
    cfg.device = 'cpu'
    cfg.image_size = 64
    cfg.use_multiscale_fusion = False
    cfg.use_real_task_heads = False

    pipeline = UWOCSemComPipeline(cfg)
    x = torch.randn(1, 3, 64, 64)
    out = pipeline(x, link_distance=10.0, phase='phase1')

    assert out['ber'].item() == 0.0
    print('test_pipeline_phase1_bypasses_channel OK')


if __name__ == '__main__':
    test_pipeline_forward_lightweight()
    test_pipeline_forward_yolo26_asff_gradient()
    test_pipeline_forward_real_heads_small_image()
    test_pipeline_phase1_bypasses_channel()
