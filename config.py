"""
Configuration globale du pipeline.
"""
import torch

class Config:
    def __init__(self):
        # Chemin vers le dataset fusionné (sortie de
        # coral_dataset_merge_pipeline_v2.ipynb) -- structure attendue :
        #   data_root/train/images/*.jpg, data_root/train/masks/*.png,
        #   data_root/train/labels.csv (filename,class_id,has_mask)
        # None ou chemin inexistant -> main.py bascule automatiquement sur
        # DummyCoralDataset (données synthétiques, pour tester le pipeline
        # sans dataset réel).
        self.data_root = r'C:\Users\smara\OneDrive\Bureau\USC\Datasets\Roboflow\merged_dataset'

        # Paramètres de quantification
        self.levels = [8, 8, 8, 6, 5]  # Configuration 14 bits nominale
        self.d = len(self.levels)      # Dimension latente (d <= 8)

        # Modèle d'extraction (émetteur)
        self.backbone_name = 'simple_cnn'  # Options : yolo26, segformer_mitb5, simple_cnn
        self.yolo_weights = 'yolo26n-seg.pt'
        self.local_weights_dir = None      # déploiement offline edge
        self.segformer_pretrained_name = None
        self.backbone_pretrained = True
        self.backbone_freeze = False

        # Fusion multi-échelle ASFF (avant le Goulot d'Étranglement Edge)
        self.use_multiscale_fusion = True  # False = mono-échelle (feature_scale=-1)
        self.fusion_channels = 64
        self.fusion_reference_index = -1   # -1 = échelle profonde, coût edge minimal

        # Têtes de tâches aval (§3.4) -- opèrent sur x_hat, indépendantes du
        # backbone émetteur. ATTENTION MÉMOIRE : True instancie 2 modèles
        # MiT-B5 complets (~80M params chacun) -- sans problème sur GPU dev
        # (RTX 4000...), mais peut nécessiter plus de RAM qu'un petit poste CPU.
        self.use_real_task_heads = True
        self.segformer_head_pretrained_name = None
        self.freeze_task_head_encoders = False
        # 4 classes -- 0=fond, 1=Healthy, 2=Dead_Coral, 3=Bleached_Coral,
        # cohérent avec le rasterisation des masques dans
        # coral_dataset_merge_pipeline_v2.ipynb (Step 3, CLASS_TO_ID).
        # ATTENTION : mettre 2 ici (ancien défaut "corail vs fond") ferait
        # planter cross_entropy dès qu'un masque contient la valeur 3
        # (IndexError: Target out of bounds) -- vérifié.
        self.num_classes_seg = 4
        self.num_classes_cls = 3          # Healthy / Dead_Coral / Bleached_Coral

        # Image
        self.image_size = 128

        # Paramètres du Canal Optique IM/DD (§3.2)
        self.channel_type = 'uwoc_im_dd'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.link_distance = 10.0  # Distance en mètres

        self.c_e = 0.151            # Atténuation Beer-Lambert (eau claire, m^-1)
        # Variance turbulence log-normale avec moyennage d'ouverture (Dr = 75 mm, Eq. 43) :
        # 1.1e-7 au lieu de 0.01/0.05 surestimé
        self.sigma_T2 = 1.1e-7
        self.use_dynamic_eq43 = False # True pour calcul Eq. 43 à la volée
        self.D_r = 0.075            # Diamètre lentille réceptrice (75 mm)
        self.theta_div = 0.05       # Demi-angle de divergence du faisceau (rad, w_L ~ theta_div * L)
        self.w0 = 20e-3 * (2.0 ** 0.5) # Waist du faisceau au plan source (m)
        self.sigma_theta_t = 2e-3   # Gigue angulaire de pointage (rad) -- §3.2
        self.P_tx = 0.01            # Puissance optique émise (W, ex 10 mW)

        # Paramètres SiPM / TIA et débit binaire
        self.R_b = 1e6                # Débit binaire (1 Mbps = 1e6 bit/s)
        self.B_e = self.R_b / 2.0     # Bande passante LPF électrique (Be ≈ Rb / 2 = 0.5 MHz)
        self.R_e = 0.8               # Responsivité SiPM (A/W)
        self.R_L = 1e3                # Résistance TIA (ohm)
        self.G = 1e5                  # Gain SiPM
        self.T_e = 290.0              # Température de bruit (K)

        # Entraînement
        self.batch_size = 4
        self.lr = 1e-4

        self.logger = None
