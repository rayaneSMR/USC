"""
Utilitaires de compression et de profilage pour l'émetteur, destinés à
évaluer/réduire le coût réel (paramètres, FLOPs, taille mémoire) sur une
cible edge (sonde sous-marine).

Ce module traite deux manques identifiés :
- point 4 : compression du backbone (YOLO26 / SegFormer) et des composants
  décodeurs via quantization statique PT2E (PyTorch 2 Export) ou dynamique,
  et pruning non-structuré.
- point 5 : mesure concrète de FLOPs/paramètres/latence pour vérifier que
  la chaîne tient dans un budget edge donné.

Compression :
- PT2E Static Quantization : utilise le flux moderne de capture de graphe
  ATen IR (`torch.export.export`), `prepare_pt2e`, observation sur données
  de calibration, et `convert_pt2e` avec `XNNPACKQuantizer`. Ce flux résout
  nativement les limitations de l'ancien Eager Mode sur YOLO26 (comme les
  additions résiduelles et activations SiLU).
- Quantization dynamique : native PyTorch pour couches Linear (SegFormer).
- Pruning non-structuré L1 : réduction des paramètres non-nuls.

Profilage : comptage des FLOPs/MACs via forward hooks Conv2d/Linear sans
dépendance externe lourde.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from torch.export import export, Dim

# Imports PT2E (avec rétrocompatibilité PyTorch 2.x / torchao)
try:
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
except (ImportError, ModuleNotFoundError):
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.testing.pt2e._xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    import sys
    import types
    if 'torch.ao.quantization.quantize_pt2e' not in sys.modules:
        _m1 = types.ModuleType('torch.ao.quantization.quantize_pt2e')
        _m1.prepare_pt2e = prepare_pt2e
        _m1.convert_pt2e = convert_pt2e
        sys.modules['torch.ao.quantization.quantize_pt2e'] = _m1
    if 'torch.ao.quantization.quantizer.xnnpack_quantizer' not in sys.modules:
        _m2 = types.ModuleType('torch.ao.quantization.quantizer.xnnpack_quantizer')
        _m2.XNNPACKQuantizer = XNNPACKQuantizer
        _m2.get_symmetric_quantization_config = get_symmetric_quantization_config
        sys.modules['torch.ao.quantization.quantizer.xnnpack_quantizer'] = _m2


# ----------------------------------------------------------------------
# Point 5 : comptage FLOPs / paramètres
# ----------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    """
    Mesure le nombre de paramètres et la taille sérialisée en Mo du modèle.
    Pour les modèles quantifiés sous PT2E ou Eager Mode, la taille du
    state_dict sérialisé reflète fidèlement la réduction de mémoire.
    """
    import io
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_mb = len(buf.getvalue()) / (1024 ** 2)

    return {'total_params': total, 'trainable_params': trainable, 'size_mb': round(size_mb, 3)}


def count_flops(model: nn.Module, input_shape: tuple, device='cpu') -> dict:
    """
    Compte les MACs (multiply-accumulate) des couches Conv2d/Linear via des
    forward hooks. FLOPs ≈ 2 × MACs (une addition + une multiplication par MAC).

    input_shape: (C, H, W) -- sans la dimension batch, ex (3, 128, 128).
    """
    macs = {'total': 0}
    hooks = []

    def conv_hook(module, inputs, output):
        out_h, out_w = output.shape[-2:]
        kh, kw = module.kernel_size
        in_c_per_group = module.in_channels // module.groups
        out_c = module.out_channels
        macs_layer = out_c * in_c_per_group * kh * kw * out_h * out_w
        macs['total'] += macs_layer

    def linear_hook(module, inputs, output):
        macs_layer = module.in_features * module.out_features
        n_elements = inputs[0].numel() // module.in_features
        macs['total'] += macs_layer * n_elements

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, *input_shape, device=device)
        try:
            model(dummy)
        finally:
            for h in hooks:
                h.remove()

    return {
        'macs': macs['total'],
        'flops': macs['total'] * 2,
        'gflops': round(macs['total'] * 2 / 1e9, 4),
    }


def profile_model(model: nn.Module, input_shape: tuple, device='cpu', name='model') -> dict:
    """Rapport combiné params + FLOPs, pratique pour vérifier un budget edge."""
    params = count_parameters(model)
    try:
        flops = count_flops(model, input_shape, device=device)
    except Exception as e:
        flops = {'error': str(e)}
    report = {'name': name, **params, **flops}
    return report


# ----------------------------------------------------------------------
# Point 4 : compression du backbone (quantization / pruning)
# ----------------------------------------------------------------------

def quantize_backbone_dynamic(model: nn.Module) -> nn.Module:
    """
    Quantization dynamique INT8 des couches nn.Linear du modèle (ex les
    projections d'attention de SegFormer). Ne quantifie PAS les Conv2d
    (majoritaires dans YOLO26). Conservée pour SegFormer (Linear-heavy)
    ou comme référence rapide. CPU uniquement.
    """
    return torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear}, dtype=torch.qint8
    )


# ----------------------------------------------------------------------
# Point 4 (suite) : quantization STATIQUE calibrée via PyTorch 2 Export (PT2E)
# ----------------------------------------------------------------------

def quantize_backbone_static(model: nn.Module, calibration_data,
                            input_shape: tuple = (3, 128, 128)) -> nn.Module:
    """
    Quantization statique INT8 post-entraînement basée sur l'API PyTorch 2 Export (PT2E).

    Contrairement à l'ancien Eager Mode, PT2E capture le graphe complet en dialecte
    ATen IR (`torch.export.export`), ce qui permet de quantifier efficacement les
    convolutions, les activations SiLU et les embranchements résiduels de YOLO26
    sans nécessiter de monkey-patching d'opérateurs internes.

    Étapes du flux PT2E :
      1. Génération d'example_inputs factice basé sur input_shape.
      2. Export du modèle en graphe ATen IR via `torch.export.export`.
      3. Initialisation d'un `XNNPACKQuantizer` avec configuration symétrique.
      4. Insertion des observateurs via `prepare_pt2e`.
      5. Calibration sur `calibration_data` (sans `torch.no_grad()`, PT2E observant
         directement les activations).
      6. Conversion finale des opérations et poids en int8 via `convert_pt2e`.

    Args:
        model: Modèle PyTorch (`nn.Module`) à quantifier.
        calibration_data: Itérable fournissant les tenseurs de calibration.
        input_shape: Forme d'entrée (C, H, W) sans la dimension batch.

    Returns:
        GraphModule PyTorch quantifié en INT8.
    """
    model = model.cpu().eval()

    # 1. example_inputs factice (avec dimension batch dynamique si supportée)
    batch_dim = Dim('batch', min=1)
    example_inputs = (torch.randn(2, *input_shape),)
    try:
        exported_program = export(model, example_inputs, dynamic_shapes=({0: batch_dim},))
    except Exception:
        example_inputs = (torch.randn(1, *input_shape),)
        exported_program = export(model, example_inputs)

    # 2. Extraction du GraphModule ATen IR
    gm = exported_program.module() if hasattr(exported_program, 'module') else exported_program

    # 3. Initialisation du quantizer XNNPACK avec configuration symétrique
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())

    # 4. Préparation PT2E (insertion des observateurs)
    prepared_model = prepare_pt2e(gm, quantizer)

    # 5. Calibration (sans torch.no_grad(), PT2E gère l'observation)
    for batch in calibration_data:
        if isinstance(batch, (list, tuple)):
            prepared_model(*batch)
        else:
            prepared_model(batch)

    # 6. Conversion en graphe INT8 quantifié
    quantized_model = convert_pt2e(prepared_model)

    # Élimination des poids float32 d'origine non référencés par le graphe quantifié
    # afin que la sérialisation / state_dict reflète fidèlement le gain mémoire int8
    used_attrs = {node.target for node in quantized_model.graph.nodes if node.op == 'get_attr'}
    for name, mod in list(quantized_model.named_modules()):
        for p_name, _ in list(mod.named_parameters(recurse=False)):
            full_name = f"{name}.{p_name}" if name else p_name
            if full_name not in used_attrs:
                delattr(mod, p_name)

    return quantized_model


def make_random_calibration_batches(input_shape: tuple, num_batches: int = 8, batch_size: int = 4):
    """
    Génère des batches aléatoires pour tester le flux de quantization statique
    en l'absence de vraies données d'imagerie sous-marine.
    """
    for _ in range(num_batches):
        yield torch.randn(batch_size, *input_shape)


def prune_backbone_unstructured(model: nn.Module, amount: float = 0.3,
                                 make_permanent: bool = True) -> nn.Module:
    """
    Pruning non structuré L1 sur les poids des couches Conv2d/Linear
    (amount = fraction des poids mis à zéro par couche). Réduit l'empreinte
    mémoire de stockage des poids.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch_prune.l1_unstructured(module, name='weight', amount=amount)
            if make_permanent:
                torch_prune.remove(module, 'weight')
    return model
