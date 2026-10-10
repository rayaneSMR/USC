"""
Échantillonneur stratifié (Stratified Sampler / Stratified Batch Sampler) pour PyTorch.
Garantit qu'au moins 50% (ou ratio configurable) de chaque batch contient des
échantillons avec de vrais masques de segmentation (has_mask=True).
Évite l'instabilité du gradient où la perte Dice s'effondre à zéro sur les batchs
purement de classification (has_mask=False).
"""
import math
import random
from typing import Iterator, List, Optional
import torch
from torch.utils.data import Sampler


def get_dataset_mask_flags(dataset) -> List[bool]:
    """Extrait la liste booléenne has_mask pour chaque indice du dataset."""
    if hasattr(dataset, 'has_mask_list') and dataset.has_mask_list is not None:
        return [bool(m) for m in dataset.has_mask_list]
    
    # Cas Subset PyTorch
    if isinstance(dataset, torch.utils.data.Subset):
        parent_flags = get_dataset_mask_flags(dataset.dataset)
        return [parent_flags[i] for i in dataset.indices]
    
    # Cas CoralDataset via filenames et masks
    if hasattr(dataset, 'filenames') and hasattr(dataset, 'mask_dir'):
        import os
        flags = []
        for fname in dataset.filenames:
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(dataset.mask_dir, stem + '.png')
            flags.append(os.path.exists(mask_path))
        return flags
    
    # Inspection directe par échantillon pour tous les indices (sans extrapolation aléatoire)
    n = len(dataset)
    flags = []
    for i in range(n):
        sample = dataset[i]
        hm = sample.get('has_mask', True)
        if isinstance(hm, torch.Tensor):
            hm = bool(hm.item())
        flags.append(bool(hm))
    return flags


class StratifiedBatchSampler(Sampler[List[int]]):
    """
    BatchSampler garantissant qu'au moins min_pos_ratio (par défaut 50%) de
    chaque batch est composé d'échantillons avec has_mask=True.
    """

    def __init__(self, dataset, batch_size: int = 4, min_pos_ratio: float = 0.5,
                 shuffle: bool = True, drop_last: bool = False, seed: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.min_pos_ratio = min_pos_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed) if seed is not None else random

        mask_flags = get_dataset_mask_flags(dataset)
        self.pos_indices = [i for i, m in enumerate(mask_flags) if m]
        self.neg_indices = [i for i, m in enumerate(mask_flags) if not m]

        # Nombre minimal d'échantillons avec masque par batch
        if len(self.pos_indices) > 0:
            self.n_pos_per_batch = min(self.batch_size, max(1, math.ceil(self.batch_size * self.min_pos_ratio)))
        else:
            self.n_pos_per_batch = 0
        self.n_neg_per_batch = self.batch_size - self.n_pos_per_batch

        # Nombre total de batches par époque
        total_samples = len(dataset)
        if self.drop_last:
            self.num_batches = total_samples // self.batch_size
        else:
            self.num_batches = math.ceil(total_samples / self.batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Cas dégradé 1 : aucun échantillon négatif (100% avec masques)
        if len(self.neg_indices) == 0:
            pos_copy = list(self.pos_indices)
            if self.shuffle:
                self.rng.shuffle(pos_copy)
            # Répéter si nécessaire pour couvrir num_batches
            idx = 0
            for _ in range(self.num_batches):
                batch = []
                while len(batch) < self.batch_size:
                    if idx >= len(pos_copy):
                        if self.shuffle:
                            self.rng.shuffle(pos_copy)
                        idx = 0
                    batch.append(pos_copy[idx])
                    idx += 1
                yield batch
            return

        # Cas dégradé 2 : aucun échantillon positif
        if len(self.pos_indices) == 0:
            neg_copy = list(self.neg_indices)
            if self.shuffle:
                self.rng.shuffle(neg_copy)
            idx = 0
            for _ in range(self.num_batches):
                batch = []
                while len(batch) < self.batch_size:
                    if idx >= len(neg_copy):
                        if self.shuffle:
                            self.rng.shuffle(neg_copy)
                        idx = 0
                    batch.append(neg_copy[idx])
                    idx += 1
                yield batch
            return

        # Cas normal : mélange des deux classes avec au moins 50% de positifs par batch
        pos_pool = list(self.pos_indices)
        neg_pool = list(self.neg_indices)
        if self.shuffle:
            self.rng.shuffle(pos_pool)
            self.rng.shuffle(neg_pool)

        pos_idx = 0
        neg_idx = 0

        for _ in range(self.num_batches):
            batch_pos = []
            for _ in range(self.n_pos_per_batch):
                if pos_idx >= len(pos_pool):
                    if self.shuffle:
                        self.rng.shuffle(pos_pool)
                    pos_idx = 0
                batch_pos.append(pos_pool[pos_idx])
                pos_idx += 1

            batch_neg = []
            for _ in range(self.n_neg_per_batch):
                if neg_idx >= len(neg_pool):
                    if self.shuffle:
                        self.rng.shuffle(neg_pool)
                    neg_idx = 0
                batch_neg.append(neg_pool[neg_idx])
                neg_idx += 1

            batch = batch_pos + batch_neg
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch


class StratifiedSampler(Sampler[int]):
    """
    Sampler PyTorch standard (renvoie un flux d'indices individuels)
    structuré de façon à ce que n'importe quel découpage séquentiel en tranches
    de batch_size contienne au moins min_pos_ratio (50%) de masques réels.
    Compatible avec DataLoader(dataset, batch_size=B, sampler=StratifiedSampler(dataset, batch_size=B)).
    """

    def __init__(self, dataset, batch_size: int = 4, min_pos_ratio: float = 0.5,
                 shuffle: bool = True, drop_last: bool = False, seed: Optional[int] = None):
        self.batch_sampler = StratifiedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            min_pos_ratio=min_pos_ratio,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        self.batch_size = self.batch_sampler.batch_size

    def __len__(self) -> int:
        return len(self.batch_sampler) * self.batch_size

    def __iter__(self) -> Iterator[int]:
        for batch in self.batch_sampler:
            for idx in batch:
                yield idx
