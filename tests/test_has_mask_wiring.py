"""
Vérifie le branchement has_mask (dataset fusionné mixte segmentation/
classification pure, voir coral_dataset_merge_pipeline_v2.ipynb) :
- CoralDataset détecte correctement has_mask via l'existence du fichier
  .png (indépendamment de l'extension .jpg de l'image, bug corrigé).
- compute_phase4_loss n'utilise le masque QUE sur has_mask=True (gradient
  nul sur les échantillons has_mask=False).
"""
import os
import csv
import numpy as np
import torch
from PIL import Image

from datasets.coral_dataset import CoralDataset
from losses.latent_align import compute_phase4_loss


def _make_fixture(root, split='train'):
    split_dir = os.path.join(root, split)
    os.makedirs(f'{split_dir}/images', exist_ok=True)
    os.makedirs(f'{split_dir}/masks', exist_ok=True)
    with open(f'{split_dir}/labels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'class_id', 'has_mask'])
        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(f'{split_dir}/images/with_mask.jpg')
        Image.fromarray(np.random.randint(0, 2, (32, 32), dtype=np.uint8)).save(f'{split_dir}/masks/with_mask.png')
        w.writerow(['with_mask.jpg', 1, 1])

        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(f'{split_dir}/images/no_mask.jpg')
        w.writerow(['no_mask.jpg', 0, 0])


def test_coral_dataset_has_mask_detection(tmp_path='/tmp/test_has_mask_fixture'):
    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)
    _make_fixture(tmp_path, split='train')

    ds = CoralDataset(root=tmp_path, split='train', image_size=32, backbone_name='simple_cnn')
    items = {ds[i]['filename']: ds[i] for i in range(len(ds))}

    assert items['with_mask.jpg']['has_mask'].item() is True
    assert items['no_mask.jpg']['has_mask'].item() is False
    # le masque placeholder (has_mask=False) doit être tout à zéro
    assert items['no_mask.jpg']['mask'].unique().tolist() == [0]
    print('test_coral_dataset_has_mask_detection OK')


def test_phase4_loss_zero_gradient_on_unmasked_samples():
    seg_logits = torch.randn(4, 3, 16, 16, requires_grad=True)
    seg_masks = torch.randint(0, 3, (4, 16, 16))
    seg_masks[2:] = 0  # placeholder pour les échantillons has_mask=False
    cls_logits = torch.randn(4, 3, requires_grad=True)
    cls_labels = torch.randint(0, 3, (4,))
    has_mask = torch.tensor([True, True, False, False])

    loss = compute_phase4_loss(seg_logits, seg_masks, cls_logits, cls_labels, has_mask=has_mask)
    loss.backward()

    assert seg_logits.grad[2:].abs().sum().item() == 0.0, "gradient devrait être nul sur has_mask=False"
    assert seg_logits.grad[:2].abs().sum().item() > 0.0, "gradient devrait être non-nul sur has_mask=True"
    print('test_phase4_loss_zero_gradient_on_unmasked_samples OK')


def test_phase4_loss_all_unmasked_batch_no_crash():
    seg_logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    seg_masks = torch.zeros(2, 8, 8, dtype=torch.long)
    cls_logits = torch.randn(2, 3, requires_grad=True)
    cls_labels = torch.randint(0, 3, (2,))
    has_mask = torch.tensor([False, False])

    loss = compute_phase4_loss(seg_logits, seg_masks, cls_logits, cls_labels, has_mask=has_mask)
    loss.backward()  # ne doit pas planter (division par zéro évitée)
    print('test_phase4_loss_all_unmasked_batch_no_crash OK')


def test_phase4_loss_backward_compat_no_has_mask():
    seg_logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    seg_masks = torch.randint(0, 3, (2, 8, 8))
    loss = compute_phase4_loss(seg_logits, seg_masks)  # has_mask=None
    loss.backward()
    print('test_phase4_loss_backward_compat_no_has_mask OK')


if __name__ == '__main__':
    test_coral_dataset_has_mask_detection()
    test_phase4_loss_zero_gradient_on_unmasked_samples()
    test_phase4_loss_all_unmasked_batch_no_crash()
    test_phase4_loss_backward_compat_no_has_mask()
