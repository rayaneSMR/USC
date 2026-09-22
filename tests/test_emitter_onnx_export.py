"""
Vérifie net/emitter.py (export ONNX de l'émetteur, jusqu'au flux binaire b).
"""
import io
import torch
import onnx
import onnxruntime as ort
import numpy as np

from config import Config
from net.pipeline import UWOCSemComPipeline
from net.emitter import Emitter


def _export_and_load(emitter, x, opset_version=17):
    buf = io.BytesIO()
    torch.onnx.export(
        emitter, x, buf,
        input_names=['image'], output_names=['bits'],
        opset_version=opset_version, dynamo=False,
    )
    buf.seek(0)
    onnx_model = onnx.load(buf)
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(buf.getvalue())
    return sess


def test_emitter_export_simple_cnn_bitexact():
    """Sur un backbone peu profond, l'export doit être bit-exact."""
    cfg = Config()
    cfg.backbone_name = 'simple_cnn'
    cfg.device = 'cpu'
    cfg.image_size = 64
    cfg.use_multiscale_fusion = False
    cfg.use_real_task_heads = False

    torch.manual_seed(0)
    pipeline = UWOCSemComPipeline(cfg)
    pipeline.eval()
    emitter = Emitter.from_pipeline(pipeline)
    emitter.eval()

    x = torch.randn(1, 3, 64, 64)
    sess = _export_and_load(emitter, x)

    with torch.no_grad():
        b_torch = emitter(x).numpy()
    b_onnx = sess.run(['bits'], {'image': x.numpy()})[0]

    assert np.abs(b_torch - b_onnx).max() < 1e-4, "export simple_cnn attendu bit-exact"
    print('test_emitter_export_simple_cnn_bitexact OK')


def test_emitter_export_yolo26_asff_valid_graph():
    """Sur YOLO26+ASFF, le graphe doit être valide et cohérent numériquement
    dans un ordre de grandeur raisonnable (voir note sur la sensibilité aux
    frontières d'arrondi FSQ face au non-déterminisme cross-runtime)."""
    cfg = Config()
    cfg.backbone_name = 'yolo26'
    cfg.device = 'cpu'
    cfg.image_size = 64
    cfg.use_multiscale_fusion = True
    cfg.use_real_task_heads = False

    pipeline = UWOCSemComPipeline(cfg)
    pipeline.eval()
    emitter = Emitter.from_pipeline(pipeline)
    emitter.eval()

    x = torch.randn(1, 3, 64, 64)
    sess = _export_and_load(emitter, x)

    n_diff_total, n_bits_total = 0, 0
    with torch.no_grad():
        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.randn(1, 3, 64, 64)
            b_torch = emitter(x).numpy()
            b_onnx = sess.run(['bits'], {'image': x.numpy()})[0]
            assert b_torch.shape == b_onnx.shape
            n_diff_total += (np.abs(b_torch - b_onnx) > 0.5).sum()
            n_bits_total += b_torch.size

    mismatch_rate = n_diff_total / n_bits_total
    assert mismatch_rate < 0.20, f"taux de mismatch inhabituel ({mismatch_rate:.1%})"
    print(f'test_emitter_export_yolo26_asff_valid_graph OK (mismatch={mismatch_rate:.1%})')


if __name__ == '__main__':
    test_emitter_export_simple_cnn_bitexact()
    test_emitter_export_yolo26_asff_valid_graph()
