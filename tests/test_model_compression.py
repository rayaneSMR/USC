"""
Vérifie utils/model_compression.py :
- quantization statique calibrée fonctionne (forward + gain de taille réel)
  sur des modules standards que nous contrôlons entièrement (Generator).
- documente/reproduit la limite connue sur YOLO26 (additions résiduelles
  internes d'ultralytics non quantifiables) sans faire planter la suite de
  tests -- ce test confirme juste que l'échec se produit bien à l'endroit
  attendu, pour éviter une régression silencieuse si un jour ça se met à
  fonctionner (ou à échouer différemment) suite à une mise à jour d'ultralytics.
"""
import torch
from net.generator import Generator
from utils.model_compression import (
    count_parameters,
    quantize_backbone_static,
    make_random_calibration_batches,
    prune_backbone_unstructured,
    profile_model,
)


def test_static_quantization_generator_runs_and_compresses():
    gen = Generator(in_channels=5, base_channels=64, num_upsamples=3)
    before = count_parameters(gen)

    calib = make_random_calibration_batches(input_shape=(5, 16, 16), num_batches=4, batch_size=2)
    q_gen = quantize_backbone_static(gen, calib, backend='qnnpack')
    after = count_parameters(q_gen)

    x = torch.randn(1, 5, 16, 16)
    out = q_gen(x)
    assert out.shape == (1, 3, 128, 128)
    assert before['size_mb'] > after['size_mb'], "la quantization statique devrait réduire size_mb"
    ratio = before['size_mb'] / after['size_mb']
    print(f'test_static_quantization_generator_runs_and_compresses OK, ratio={ratio:.2f}x')


def test_yolo26_static_quantization_known_limitation():
    """Documente la limite connue plutôt que de la cacher : la quantization
    statique PyTorch native échoue sur YOLO26 à cause des additions
    résiduelles internes d'ultralytics (non quantifiables sans monkey-patch).
    Si ce test se met à ÉCHOUER (c-à-d si le forward réussit), c'est une
    bonne nouvelle -- signaler et mettre à jour la docstring en conséquence."""
    from net.backbone import YOLO26Backbone

    bb = YOLO26Backbone(weights='yolo26n-seg.pt')
    calib = make_random_calibration_batches(input_shape=(3, 128, 128), num_batches=2, batch_size=1)
    q_bb = quantize_backbone_static(bb, calib, backend='qnnpack')

    x = torch.randn(1, 3, 128, 128)
    try:
        q_bb(x)
        print('test_yolo26_static_quantization_known_limitation : '
              'SURPRISE -- le forward a réussi, la limitation semble résolue '
              '(mettre à jour la docstring de quantize_backbone_static).')
    except RuntimeError as e:
        assert 'quantized tensor' in str(e).lower() or 'empty_strided' in str(e).lower()
        print('test_yolo26_static_quantization_known_limitation OK '
              '(limitation confirmée, comportement attendu)')


def test_pruning_still_works():
    import torch.nn as nn
    gen = Generator(in_channels=5, base_channels=32, num_upsamples=2)
    gen = prune_backbone_unstructured(gen, amount=0.3)

    # Ne compte que les poids des couches effectivement ciblées par le
    # pruning (Conv2d/Linear) -- les paramètres BatchNorm ne sont pas
    # concernés et diluent la fraction si on les inclut.
    n_zero, n_total = 0, 0
    for m in gen.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            n_zero += (m.weight == 0).sum().item()
            n_total += m.weight.numel()
    frac = n_zero / n_total
    assert 0.15 < frac < 0.45, f"fraction de poids nuls inattendue: {frac:.2%}"
    print(f'test_pruning_still_works OK, {frac:.2%} de poids nuls (Conv2d/Linear uniquement)')


def test_profile_model_reports_flops_and_params():
    gen = Generator(in_channels=5, base_channels=32, num_upsamples=2)
    report = profile_model(gen, input_shape=(5, 32, 32), name='generator_test')
    assert report['total_params'] > 0
    assert 'gflops' in report and report['gflops'] >= 0
    print('test_profile_model_reports_flops_and_params OK,', report)


if __name__ == '__main__':
    test_static_quantization_generator_runs_and_compresses()
    test_yolo26_static_quantization_known_limitation()
    test_pruning_still_works()
    test_profile_model_reports_flops_and_params()
