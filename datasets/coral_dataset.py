"""
Dataset pour images sous-marines (coraux) + masques de segmentation +
labels de classification. Structure attendue (sortie de
coral_dataset_merge_pipeline_v2.ipynb) :
    data_root/
        images/*.jpg
        masks/*.png          (même NOM DE FICHIER que images/, extension différente --
                               ex 'foo.jpg' -> masque 'foo.png' -- uniquement pour les
                               échantillons has_mask=1)
        labels.csv            filename,class_id,has_mask

NORMALISATION SPÉCIFIQUE AU BACKBONE (point important) :
- 'yolo26' : ultralytics normalise uniquement en [0,1] (division par 255),
  SANS soustraction de moyenne ImageNet -- c'est le prétraitement natif de
  YOLO. Appliquer une normalisation ImageNet ici dégraderait les features
  extraites par un backbone pré-entraîné.
- 'segformer_mitb5' : HuggingFace/transformers attend une normalisation
  ImageNet standard (mean/std ci-dessous), cohérente avec
  `SegformerImageProcessor` par défaut.
- 'simple_cnn' : pas de contrainte (backbone non pré-entraîné), on garde [0,1].

GESTION DE has_mask (point important) : le dataset fusionné mélange des
images avec vrai masque de segmentation (sources COCO, polygones
rasterisés) et des images SANS masque (source `bleach_detection`,
classification pure). Pour ces dernières, `__getitem__` retourne un masque
de zéros ET `has_mask=0` -- CE MASQUE DE ZÉROS NE DOIT JAMAIS SERVIR DE
CIBLE DE SEGMENTATION réelle (il ne signifie PAS "tout est fond"), c'est un
simple placeholder pour permettre le batching. `losses/latent_align.py::
compute_phase4_loss` doit filtrer sur `has_mask` avant de calculer la perte
de segmentation -- ne jamais utiliser le masque d'un échantillon has_mask=0.
"""
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Backbones nécessitant une normalisation ImageNet après le ToTensor() [0,1]
_BACKBONES_REQUIRING_IMAGENET_NORM = {'segformer_mitb5'}


class CoralDataset(Dataset):
    def __init__(self, root, split='train', image_size=512, has_labels=True,
                 backbone_name='yolo26'):
        """
        root: dossier PARENT contenant train/valid/test (sortie de
              coral_dataset_merge_pipeline_v2.ipynb), ex 'merged_dataset'.
              `split` sélectionne le sous-dossier -- BUG CORRIGÉ : `split`
              était accepté mais jamais utilisé, `root` était utilisé
              directement (donc toujours 'train' implicitement, quel que
              soit `split` demandé).
        """
        self.root = root
        self.split = split
        split_dir = os.path.join(root, split)
        self.image_dir = os.path.join(split_dir, 'images')
        self.mask_dir = os.path.join(split_dir, 'masks')
        self.has_labels = has_labels
        self.backbone_name = backbone_name

        self.filenames = sorted(os.listdir(self.image_dir)) if os.path.isdir(self.image_dir) else []

        # labels.csv : filename,class_id[,has_mask]. Le format 2-colonnes
        # (sans has_mask) reste supporté pour compatibilité arrière -- dans
        # ce cas has_mask est déduit de la présence physique du fichier
        # masque correspondant.
        self.labels = {}
        self.has_mask_flags = {}
        labels_csv = os.path.join(split_dir, 'labels.csv')
        if has_labels and os.path.exists(labels_csv):
            with open(labels_csv, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                has_mask_col = header is not None and len(header) >= 3 and header[2] == 'has_mask'
                for row in reader:
                    self.labels[row[0]] = int(row[1])
                    if has_mask_col:
                        self.has_mask_flags[row[0]] = int(row[2])

        image_tf_steps = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # -> [0,1]
        ]
        if backbone_name in _BACKBONES_REQUIRING_IMAGENET_NORM:
            image_tf_steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        # yolo26/simple_cnn : on s'arrête à [0,1], pas de normalisation ImageNet.
        self.image_tf = T.Compose(image_tf_steps)

        self.mask_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        x = self.image_tf(img)

        # Le masque partage le STEM du nom de fichier mais pas forcément
        # l'extension (images en .jpg, masques toujours en .png -- voir
        # net/multiscale... non, voir coral_dataset_merge_pipeline_v2.ipynb
        # Step 3 : Image.fromarray(mask_arr, mode='L').save(..., '.png')).
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(self.mask_dir, stem + '.png')

        has_mask_flag = self.has_mask_flags.get(fname, None)
        mask_file_exists = os.path.exists(mask_path)
        # Cohérence : si labels.csv dit has_mask=1 mais le fichier est
        # introuvable (ou l'inverse), on fait confiance au fichier réel --
        # plus robuste qu'un flag CSV potentiellement désynchronisé.
        has_mask = mask_file_exists

        if has_mask:
            mask = Image.open(mask_path)
            mask = self.mask_tf(mask)
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            # Placeholder de zéros -- NE PAS interpréter comme cible réelle,
            # voir has_mask dans le dict retourné et la note en tête de fichier.
            mask = torch.zeros(x.shape[1], x.shape[2], dtype=torch.long)

        label = self.labels.get(fname, 0)

        return {
            'image': x,
            'mask': mask,
            'has_mask': torch.tensor(has_mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.long),
            'filename': fname,
        }


class DummyCoralDataset(Dataset):
    """
    Dataset synthétique (images/masques/labels aléatoires) -- permet de
    faire tourner tout le pipeline et les 4 phases d'entraînement de bout
    en bout SANS données réelles (le branchement du vrai dataset CoralMask
    fusionné est traité séparément). Utile pour valider l'assemblage
    complet (main.py, engine/*), pas pour obtenir des résultats
    scientifiques valides.
    """

    def __init__(self, num_samples=64, image_size=128, num_classes_seg=2, num_classes_cls=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes_seg = num_classes_seg
        self.num_classes_cls = num_classes_cls

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, self.image_size, self.image_size)
        mask = torch.randint(0, self.num_classes_seg, (self.image_size, self.image_size), dtype=torch.long)
        label = torch.randint(0, self.num_classes_cls, (1,), dtype=torch.long).squeeze(0)
        return {
            'image': image,
            'mask': mask,
            'has_mask': torch.tensor(True),  # toujours True : données synthétiques, pas de contrainte réelle
            'label': label,
            'filename': f'dummy_{idx}.jpg',
        }
