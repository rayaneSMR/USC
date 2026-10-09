"""
Tests unitaires pour la gestion et l'écrasement des configurations YAML dans main.py :
- Chargement des configurations 5 bits, 10 bits, 14 bits et 16 bits
- Écrasement effectif de cfg.levels et de la dimension latente cfg.d
- Respect des paramètres de canal, distance et entraînement
"""
import os
import torch
from config import Config
from main import apply_yaml_config
from net.pipeline import UWOCSemComPipeline


def test_yaml_config_5bit():
    cfg = Config()
    assert cfg.levels == [8, 8, 8, 6, 5]  # défaut initial 14-bit
    assert cfg.d == 5

    yaml_path = 'configs_yaml/codebook_5bit.yaml'
    cfg = apply_yaml_config(cfg, yaml_path)

    assert cfg.levels == [2, 2, 2, 2, 2], f"Échec levels 5-bit: {cfg.levels}"
    assert cfg.d == 5, f"Échec dimension d: {cfg.d}"
    assert cfg.link_distance == 15.0
    print("test_yaml_config_5bit OK (levels=[2,2,2,2,2], d=5)")


def test_yaml_config_10bit():
    cfg = Config()
    yaml_path = 'configs_yaml/codebook_10bit.yaml'
    cfg = apply_yaml_config(cfg, yaml_path)

    assert cfg.levels == [8, 5, 5, 5], f"Échec levels 10-bit: {cfg.levels}"
    assert cfg.d == 4, f"Échec dimension d: {cfg.d}"
    assert cfg.link_distance == 12.0
    print("test_yaml_config_10bit OK (levels=[8,5,5,5], d=4)")


def test_yaml_config_16bit():
    cfg = Config()
    yaml_path = 'configs_yaml/codebook_16bit.yaml'
    cfg = apply_yaml_config(cfg, yaml_path)

    assert cfg.levels == [8, 8, 8, 5, 5, 5], f"Échec levels 16-bit: {cfg.levels}"
    assert cfg.d == 6, f"Échec dimension d: {cfg.d}"
    assert cfg.link_distance == 8.0
    print("test_yaml_config_16bit OK (levels=[8,8,8,5,5,5], d=6)")


def test_pipeline_instantiation_with_5bit_and_10bit():
    """
    Vérifie que UWOCSemComPipeline s'instancie et fait un forward pass correct
    avec les configurations 5-bit et 10-bit chargées depuis YAML.
    """
    # 5 bits
    cfg_5 = apply_yaml_config(Config(), 'configs_yaml/codebook_5bit.yaml')
    cfg_5.use_real_task_heads = False  # stubs pour test rapide
    cfg_5.image_size = 64
    model_5 = UWOCSemComPipeline(cfg_5)
    x = torch.rand(2, 3, cfg_5.image_size, cfg_5.image_size)
    out_5 = model_5(x, link_distance=cfg_5.link_distance, phase='phase1')
    assert out_5['x_hat'].shape == x.shape
    assert model_5.fsq.d == 5

    # 10 bits
    cfg_10 = apply_yaml_config(Config(), 'configs_yaml/codebook_10bit.yaml')
    cfg_10.use_real_task_heads = False  # stubs pour test rapide
    cfg_10.image_size = 64
    model_10 = UWOCSemComPipeline(cfg_10)
    out_10 = model_10(x, link_distance=cfg_10.link_distance, phase='phase1')
    assert out_10['x_hat'].shape == x.shape
    assert model_10.fsq.d == 4

    print("test_pipeline_instantiation_with_5bit_and_10bit OK")


if __name__ == '__main__':
    test_yaml_config_5bit()
    test_yaml_config_10bit()
    test_yaml_config_16bit()
    test_pipeline_instantiation_with_5bit_and_10bit()
