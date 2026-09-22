"""
Utilitaires de compression et de profilage pour l'émetteur, destinés à
évaluer/réduire le coût réel (paramètres, FLOPs, taille mémoire) sur une
cible edge (sonde sous-marine).

Ce module traite deux manques identifiés :
- point 4 : rien ne compressait le backbone lui-même (YOLO26/SegFormer
  restent en float32 pleine précision ; seule la SORTIE est quantifiée
  via FSQ, pas le modèle).
- point 5 : aucune mesure concrète de FLOPs/paramètres/latence pour
  vérifier que la chaîne tient dans un budget edge donné.

Compression : on utilise la quantization dynamique native PyTorch
(`torch.quantization.quantize_dynamic`), qui ne nécessite ni calibration
ni ré-entraînement -- adaptée aux couches Linear (attention de SegFormer)
mais avec un effet limité sur les Conv2d (majoritaires dans YOLO26 et les
premières couches SegFormer), qui nécessiteraient une quantization statique
avec calibration pour un vrai gain. On expose aussi un pruning non structuré
simple (`torch.nn.utils.prune`) comme alternative légère et complémentaire.

Profilage : comptage manuel des FLOPs/MACs via des hooks forward sur les
couches Conv2d/Linear -- volontairement sans dépendance externe (thop,
fvcore, etc.) pour rester simple et transparent.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune


# ----------------------------------------------------------------------
# Point 5 : comptage FLOPs / paramètres
# ----------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    """
    NOTE : pour un modèle quantifié (voir `quantize_backbone_static`), les
    poids sont packés dans des tenseurs quantized internes qui ne sont PAS
    de simples `nn.Parameter` -- `numel()*element_size()` sur
    `model.parameters()` sous-estime alors fortement (voire ignore) leur
    taille réelle. `size_mb` mesure donc la taille RÉELLEMENT sérialisée
    du `state_dict` (fiable dans tous les cas, quantifié ou non), plutôt
    que de recalculer une taille théorique à partir des seuls paramètres
    itérables.
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
        # tient compte du nombre d'éléments traités en amont (ex tokens)
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
    [LIMITÉ -- voir quantize_backbone_static ci-dessous pour un vrai gain]

    Quantization dynamique INT8 des couches nn.Linear du modèle (ex les
    projections d'attention de SegFormer). Ne quantifie PAS les Conv2d
    (majoritaires dans YOLO26) : la quantization dynamique PyTorch native
    ne les supporte pas -- résultat quasi nul sur un backbone convolutif.
    Conservée pour SegFormer (Linear-heavy) ou comme référence rapide.
    CPU uniquement.
    """
    return torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear}, dtype=torch.qint8
    )


# ----------------------------------------------------------------------
# Point 4 (suite) : quantization STATIQUE calibrée -- gain réel sur Conv2d
# ----------------------------------------------------------------------

class _QuantWrapper(nn.Module):
    """Enveloppe un module quelconque avec QuantStub/DeQuantStub, requis par
    le flux de quantization statique PyTorch (eager mode) : convertit
    l'entrée float -> int8 avant le modèle, et la sortie int8 -> float après.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.module = module
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.module(x)
        x = self.dequant(x)
        return x


class _DequantActQuant(nn.Module):
    """
    Enveloppe une activation non supportée par le backend quantized CPU
    (ex nn.SiLU, utilisée par YOLO26 -- confirmé empiriquement : la
    quantization statique naïve plante à l'inférence avec
    `NotImplementedError: Could not run 'aten::silu.out' ... QuantizedCPU`)
    en la dé-quantifiant juste avant, puis en re-quantifiant juste après.
    Technique standard de la quantization eager-mode PyTorch pour les
    opérateurs sans noyau int8 natif : l'activation elle-même continue de
    tourner en float32 (coût de calcul additionnel négligeable, une seule
    activation par bloc), mais les Conv2d environnantes restent quantifiées.
    """

    def __init__(self, act_module: nn.Module):
        super().__init__()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.act = act_module
        self.quant = torch.ao.quantization.QuantStub()

    def forward(self, x):
        x = self.dequant(x)
        x = self.act(x)
        x = self.quant(x)
        return x


def _replace_unsupported_activations(model: nn.Module, unsupported_types=(nn.SiLU,)) -> int:
    """Remplace récursivement chaque instance d'un type d'activation non
    quantifiable par `_DequantActQuant`. Retourne le nombre de remplacements
    effectués (0 = rien à faire, le modèle ne contient pas ces activations)."""
    count = 0
    for name, child in model.named_children():
        if isinstance(child, unsupported_types):
            setattr(model, name, _DequantActQuant(child))
            count += 1
        else:
            count += _replace_unsupported_activations(child, unsupported_types)
    return count


def quantize_backbone_static(model: nn.Module, calibration_data, backend: str = 'qnnpack',
                              unsupported_activation_types=(nn.SiLU,)) -> nn.Module:
    """
    Quantization POST-TRAINING STATIQUE INT8, calibrée sur des données
    réelles -- contrairement à `quantize_backbone_dynamic`, celle-ci
    quantifie AUSSI les Conv2d (donc a un effet réel sur YOLO26/CNN) et
    quantifie aussi les activations, pas seulement les poids.

    Étapes du flux PyTorch eager-mode :
      1. Enveloppe le modèle avec QuantStub/DeQuantStub.
      2. Assigne une `qconfig` (schéma d'observation des plages de valeurs).
      3. `prepare()` insère des observateurs dans chaque couche quantifiable.
      4. CALIBRATION : fait tourner le modèle en mode eval sur des données
         réelles représentatives -- les observateurs enregistrent la plage
         (min/max) réelle de chaque activation. SANS CETTE ÉTAPE avec de
         vraies données, les facteurs d'échelle (scale/zero_point) sont mal
         calibrés et la précision s'effondre. Des données aléatoires en
         remplacement (si `calibration_data` est absent) donnent un résultat
         non représentatif -- utile seulement pour vérifier que le flux
         s'exécute sans erreur, PAS pour un déploiement réel.
      5. `convert()` remplace les couches float par leurs équivalents int8
         calibrés (poids ET activations).

    calibration_data: itérable de tenseurs (B, C, H, W) représentatifs des
        images réellement vues par le capteur (ex quelques centaines
        d'images de coraux). Chaque batch est simplement passé en forward
        sous `torch.no_grad()`, aucun label nécessaire (non supervisé).
    backend: 'qnnpack' pour cible ARM/edge (mobile, embarqué -- notre cas),
        'fbgemm' pour cible x86 (utile seulement pour tester/valider sur
        la machine de développement, pas pour le déploiement final edge).

    Limite connue : le flux eager-mode ci-dessous nécessite que `model`
    expose un unique `forward(x) -> tensor` standard. Notre `YOLO26Backbone`
    remplit cette condition pour `forward()` (retourne l'échelle unique
    `feature_scale`), mais PAS pour `extract_multi_scale_features()` (qui
    contourne `forward()` et lit directement le hook interne) -- cette
    seconde voie n'est pas quantifiable par ce flux.

    Note sur la quantization FX graph mode (torch.ao.quantization.quantize_fx),
    envisagée initialement comme alternative "plus robuste" : VÉRIFIÉ
    empiriquement qu'elle échoue aussi sur `YOLO26Backbone` (`TraceError:
    Proxy object cannot be iterated`), car le tracing symbolique FX ne
    supporte pas le flux de contrôle interne du modèle ultralytics ni le
    mécanisme de hook utilisé ici pour intercepter les features.

    LIMITE FINALE CONSTATÉE SUR YOLO26 (au-delà de SiLU, contourné
    ci-dessus) : les blocs résiduels internes d'ultralytics
    (`ultralytics/nn/modules/block.py`, `x + self.cv2(self.cv1(x))`)
    utilisent une addition Python native entre tenseurs, qui n'a pas de
    noyau quantized CPU (`RuntimeError: empty_strided not supported on
    quantized tensors`). Contrairement à SiLU, ce n'est PAS contournable
    par un simple remplacement de sous-module depuis l'extérieur : il
    faudrait monkey-patcher le code interne d'ultralytics pour y insérer un
    `torch.ao.nn.quantized.FloatFunctional` à la place de l'addition -- trop
    invasif et fragile (casserait à la moindre mise à jour d'ultralytics).

    CONCLUSION PRATIQUE : cette fonction fonctionne et donne un vrai gain
    mesuré (~3-4x sur la taille sérialisée) sur les modules ENTIÈREMENT
    sous notre contrôle et sans connexions résiduelles ni SiLU (ex
    `net/generator.py`, testé et validé) -- mais PAS sur le backbone YOLO26
    pré-entraîné tel quel. Pour quantifier réellement YOLO26 en vue d'un
    déploiement edge, la voie réaliste est l'export ONNX + quantization via
    ONNX Runtime, qui gère nativement les additions résiduelles et SiLU non
    supportées par insertion de nœuds QDQ au niveau du graphe -- hors du
    cadre de cet utilitaire PyTorch pur.
    """
    torch.backends.quantized.engine = backend

    model = model.cpu().eval()
    n_replaced = _replace_unsupported_activations(model, unsupported_activation_types)
    if n_replaced > 0:
        # SiLU (et toute activation listée dans `unsupported_activation_types`)
        # n'a pas de noyau int8 natif sur le backend QuantizedCPU -- confirmé
        # empiriquement (`NotImplementedError: aten::silu.out ... QuantizedCPU`).
        # On l'entoure de dequant/quant pour qu'elle continue de tourner en
        # float32 pendant que les Conv2d environnantes restent quantifiées.
        pass

    wrapped = _QuantWrapper(model)
    wrapped.qconfig = torch.ao.quantization.get_default_qconfig(backend)

    prepared = torch.ao.quantization.prepare(wrapped, inplace=False)

    prepared.eval()
    with torch.no_grad():
        for batch in calibration_data:
            prepared(batch)

    quantized = torch.ao.quantization.convert(prepared, inplace=False)
    return quantized


def make_random_calibration_batches(input_shape: tuple, num_batches: int = 8, batch_size: int = 4):
    """
    Génère des batches ALÉATOIRES pour calibrer `quantize_backbone_static`
    en l'absence de vraies données -- UNIQUEMENT pour vérifier que le flux
    de quantization s'exécute sans erreur (tests, développement). Donne une
    calibration non représentative : à ne jamais utiliser pour un modèle
    destiné à un déploiement réel, où `calibration_data` doit provenir
    d'images de coraux réelles (voir docstring `quantize_backbone_static`).
    """
    for _ in range(num_batches):
        yield torch.randn(batch_size, *input_shape)


def prune_backbone_unstructured(model: nn.Module, amount: float = 0.3,
                                 make_permanent: bool = True) -> nn.Module:
    """
    Pruning non structuré L1 sur les poids des couches Conv2d/Linear
    (amount = fraction des poids mis à zéro par couche). Réduit le nombre
    de paramètres non-nuls mais ne réduit PAS directement les FLOPs/latence
    sans support matériel/logiciel pour les tenseurs creux -- utile surtout
    pour réduire l'empreinte mémoire de stockage des poids, pas la vitesse
    d'inférence, sauf backend creux dédié.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch_prune.l1_unstructured(module, name='weight', amount=amount)
            if make_permanent:
                torch_prune.remove(module, 'weight')
    return model
