"""
Tests unitaires pour datasets/sampler.py :
- Garantie stricte qu'au moins 50% de chaque batch contient des masques réels
- Résolution de l'instabilité du gradient en Phase 4
- Gestion des cas limites (100% masqués, 0% masqués, déséquilibres extrêmes)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.sampler import StratifiedSampler, StratifiedBatchSampler, get_dataset_mask_flags
from losses.latent_align import compute_phase4_loss


class SyntheticMixedDataset(Dataset):
    def __init__(self, num_samples=100, pos_ratio=0.2):
        self.num_samples = num_samples
        self.num_pos = int(round(num_samples * pos_ratio))
        # Ex: 20 masqués (has_mask=True), 80 non masqués (has_mask=False)
        self.has_mask_list = [True] * self.num_pos + [False] * (num_samples - self.num_pos)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        has_mask = self.has_mask_list[idx]
        image = torch.rand(3, 32, 32)
        if has_mask:
            mask = torch.randint(0, 4, (32, 32), dtype=torch.long)
        else:
            mask = torch.zeros(32, 32, dtype=torch.long)
        label = torch.randint(0, 3, (1,)).squeeze()
        return {
            'image': image,
            'mask': mask,
            'has_mask': torch.tensor(has_mask, dtype=torch.bool),
            'label': label,
            'idx': idx,
        }


def test_stratified_sampler_guarantees_at_least_50_percent_masks():
    """
    Vérifie qu'avec seulement 20% d'images masquées dans le dataset global,
    100% des batches générés par le DataLoader contiennent au moins 50% d'images masquées.
    """
    dataset = SyntheticMixedDataset(num_samples=100, pos_ratio=0.2)
    batch_size = 4
    sampler = StratifiedSampler(dataset, batch_size=batch_size, min_pos_ratio=0.5, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    batches_tested = 0
    for batch in dataloader:
        has_mask = batch['has_mask']
        pos_count = has_mask.sum().item()
        pos_ratio = pos_count / len(has_mask)
        assert pos_ratio >= 0.5, f"Batch avec seulement {pos_count}/{len(has_mask)} masques ({pos_ratio:.1%}) < 50%"
        batches_tested += 1

    assert batches_tested > 0
    print(f"test_stratified_sampler_guarantees_at_least_50_percent_masks OK ({batches_tested} batches testés, tous >= 50% masques)")


def test_phase4_gradient_always_active_with_stratified_sampler():
    """
    Vérifie que la perte Dice et le gradient de segmentation ne s'annulent JAMAIS
    grâce au StratifiedSampler, même sur un dataset à 90% non-masqué.
    """
    dataset = SyntheticMixedDataset(num_samples=100, pos_ratio=0.1)
    batch_size = 6
    sampler = StratifiedSampler(dataset, batch_size=batch_size, min_pos_ratio=0.5, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    for i, batch in enumerate(dataloader):
        seg_logits = torch.randn(batch_size, 4, 32, 32, requires_grad=True)
        seg_masks = batch['mask']
        cls_logits = torch.randn(batch_size, 3, requires_grad=True)
        cls_labels = batch['label']
        has_mask = batch['has_mask']

        loss = compute_phase4_loss(
            seg_logits=seg_logits,
            seg_masks=seg_masks,
            cls_logits=cls_logits,
            cls_labels=cls_labels,
            has_mask=has_mask,
        )
        loss.backward()

        # Le gradient de segmentation DOIT être non-nul sur chaque batch
        assert seg_logits.grad is not None
        assert seg_logits.grad.abs().sum().item() > 0.0, f"Gradient nul sur l'itération {i} !"

        if i >= 5:
            break

    print("test_phase4_gradient_always_active_with_stratified_sampler OK (gradient segmentation toujours actif)")


def test_stratified_sampler_edge_cases():
    # 100% masqué
    ds_all_pos = SyntheticMixedDataset(num_samples=20, pos_ratio=1.0)
    s_all_pos = StratifiedSampler(ds_all_pos, batch_size=4)
    dl = DataLoader(ds_all_pos, batch_size=4, sampler=s_all_pos)
    for b in dl:
        assert b['has_mask'].all()

    # 0% masqué
    ds_all_neg = SyntheticMixedDataset(num_samples=20, pos_ratio=0.0)
    s_all_neg = StratifiedSampler(ds_all_neg, batch_size=4)
    dl_neg = DataLoader(ds_all_neg, batch_size=4, sampler=s_all_neg)
    for b in dl_neg:
        assert not b['has_mask'].any()

    print("test_stratified_sampler_edge_cases OK")


if __name__ == '__main__':
    test_stratified_sampler_guarantees_at_least_50_percent_masks()
    test_phase4_gradient_always_active_with_stratified_sampler()
    test_stratified_sampler_edge_cases()
