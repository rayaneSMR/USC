"""
Définitions argparse pour main.py, dans l'esprit du repo SJTU-mxtao
(args.channel_type, args.multiple_snr, etc.) mais étendu à nos 4 phases AMP.
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='UWOC-SemCom pipeline')

    parser.add_argument('--config', type=str, default='configs_yaml/codebook_14bit.yaml',
                        help='Chemin vers le fichier YAML de configuration (codebook + canal)')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4], default=None,
                        help='Phase d\'entraînement AMP: 1=FE, 2=RE, 3=TTA, 4=Task Adaptation (None = planification complète)')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--resume', type=str, default=None,
                         help='Checkpoint à charger (phase précédente ou reprise)')

    # Canal
    parser.add_argument('--channel_type', type=str, default=None,
                         choices=['none', 'uwoc_im_dd'])
    parser.add_argument('--link_distance', type=float, default=None,
                         help='Distance émetteur-récepteur en mètres')

    # Entraînement
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None,
                         help='Surcharge le nombre d\'époques de la phase courante')
    parser.add_argument('--seed', type=int, default=None)

    # Sous-étapes Phase 2
    parser.add_argument('--sub_phase', type=str, default='2A', choices=['2A', '2B'],
                         help='Utilisé uniquement si --phase 2')

    # Données et Modèle
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--use_real_task_heads', type=lambda x: str(x).lower() in ('true', '1', 'yes'), default=None,
                        help='Activer les têtes SegFormer (True) ou têtes CNN légères (False)')

    return parser.parse_args()
