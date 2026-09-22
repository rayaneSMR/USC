"""
Vérifie que les backbones réels (YOLO26-seg, SegFormer MiT-B5) s'exécutent
end-to-end, exposent out_channels correctement, et laissent passer le
gradient (nécessaire pour la Phase 2B qui met à jour l'encodeur).
"""
import torch
from net.backbone import Backbone


def test_simple_cnn_backbone():
    bb = Backbone('simple_cnn')
    x = torch.randn(1, 3, 128, 128)
    out = bb(x)
    assert out.shape[1] == bb.out_channels
    print('test_simple_cnn_backbone OK, shape=', out.shape)


def test_yolo26_backbone_shapes_and_gradient():
    bb = Backbone('yolo26', yolo_weights='yolo26n-seg.pt')
    x = torch.randn(1, 3, 128, 128, requires_grad=True)
    out = bb(x)
    assert out.shape[1] == bb.out_channels

    multi = bb.extract_multi_scale_features(x)
    assert len(multi) == 3, "YOLO26-seg attendu : 3 échelles multi-scale (P3,P4,P5)"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    print('test_yolo26_backbone_shapes_and_gradient OK, out_channels=', bb.out_channels,
          'multi-scale=', [f.shape for f in multi])


def test_segformer_backbone_shapes():
    # pretrained=False -> config MiT-B5 non pré-entraînée (pas d'accès HuggingFace requis)
    bb = Backbone('segformer_mitb5', pretrained=False)
    x = torch.randn(1, 3, 128, 128)
    out = bb(x)
    assert out.shape[1] == bb.out_channels == 512

    multi = bb.extract_multi_scale_features(x)
    assert len(multi) == 4, "SegFormer MiT-B5 attendu : 4 échelles"
    print('test_segformer_backbone_shapes OK, out_channels=', bb.out_channels,
          'multi-scale=', [f.shape for f in multi])


if __name__ == '__main__':
    test_simple_cnn_backbone()
    test_yolo26_backbone_shapes_and_gradient()
    test_segformer_backbone_shapes()
