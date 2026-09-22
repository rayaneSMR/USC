"""
Gel/dégel de sous-modules pour l'orchestration AMP (Phases 1-4).
"""


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True
    module.train()


def freeze_pipeline_parts(pipeline, part_names):
    """
    part_names: liste de noms d'attributs du SemComPipeline à geler,
    ex ['backbone', 'edge_bottleneck', 'fsq'] pour la Phase 2A.
    """
    for name in part_names:
        submodule = getattr(pipeline, name, None)
        if submodule is not None:
            freeze_module(submodule)


def unfreeze_pipeline_parts(pipeline, part_names):
    for name in part_names:
        submodule = getattr(pipeline, name, None)
        if submodule is not None:
            unfreeze_module(submodule)
