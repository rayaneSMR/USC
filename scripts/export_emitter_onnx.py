"""
Exporte l'émetteur (backbone -> fusion -> bottleneck -> FSQ -> bit mapping)
en ONNX pour déploiement edge, et vérifie la cohérence numérique.

Usage:
    python scripts/export_emitter_onnx.py --output emitter.onnx --image_size 128
"""
import argparse
import numpy as np
import onnx
import onnxruntime as ort
import torch

from config import Config
from net.pipeline import UWOCSemComPipeline
from net.emitter import Emitter


def main():
    parser = argparse.ArgumentParser(description='Export ONNX de l\'émetteur UWOC-SemCom')
    parser.add_argument('--output', type=str, default='emitter.onnx')
    parser.add_argument('--backbone_name', type=str, default='simple_cnn')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--opset_version', type=int, default=17)
    parser.add_argument('--n_check_samples', type=int, default=10)
    args = parser.parse_args()

    cfg = Config()
    cfg.backbone_name = args.backbone_name
    cfg.image_size = args.image_size
    cfg.device = 'cpu'
    cfg.use_real_task_heads = False  # inutile pour l'export émetteur seul

    pipeline = UWOCSemComPipeline(cfg)
    pipeline.eval()
    emitter = Emitter.from_pipeline(pipeline)
    emitter.eval()

    x_example = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    torch.onnx.export(
        emitter, x_example, args.output,
        input_names=['image'], output_names=['bits'],
        dynamic_axes={'image': {0: 'batch'}, 'bits': {0: 'batch'}},
        opset_version=args.opset_version, dynamo=False,
    )
    print(f'Export ONNX écrit : {args.output}')

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print('Graphe ONNX valide (onnx.checker.check_model OK)')

    sess = ort.InferenceSession(args.output)
    n_diff_total, n_bits_total = 0, 0
    with torch.no_grad():
        for _ in range(args.n_check_samples):
            x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
            b_torch = emitter(x).numpy()
            b_onnx = sess.run(['bits'], {'image': x.numpy()})[0]
            n_diff_total += (np.abs(b_torch - b_onnx) > 0.5).sum()
            n_bits_total += b_torch.size

    mismatch_rate = n_diff_total / n_bits_total if n_bits_total else 0.0
    print(f'Vérification PyTorch vs ONNX Runtime : {mismatch_rate:.2%} de bits différents.')


if __name__ == '__main__':
    main()
