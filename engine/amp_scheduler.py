"""
Alternating Multi-Phase (AMP) Scheduler.
Orchestre la transition entre les phases d'entraînement en fonction de l'époque.
"""
class AMPScheduler:
    def __init__(self, total_epochs, p1_ratio=0.2, p2_ratio=0.4, p3_ratio=0.2):
        self.total_epochs = total_epochs
        self.p1_end = int(total_epochs * p1_ratio)
        self.p2_end = self.p1_end + int(total_epochs * p2_ratio)
        self.p3_end = self.p2_end + int(total_epochs * p3_ratio)
        
    def get_phase(self, epoch):
        if epoch < self.p1_end:
            return 'phase1', "Extraction de Caractéristiques (FE)"
        elif epoch < self.p2_end:
            return 'phase2a', "Renforcement de Robustesse (MATK + BFP)"
        elif epoch < self.p3_end:
            return 'phase3', "Alignement Entraînement-Test (TTA)"
        else:
            return 'phase4', "Adaptation des Têtes de Tâches"
