"""
Alternating Multi-Phase (AMP) Scheduler.
Orchestre la transition entre les phases d'entraînement en fonction de l'époque.
"""
class AMPScheduler:
    def __init__(self, total_epochs, p1_ratio=0.2, p2_ratio=0.4, p3_ratio=0.2):
        self.total_epochs = total_epochs
        if total_epochs >= 4:
            p1_epochs = max(1, int(round(total_epochs * p1_ratio)))
            p2_epochs = max(1, int(round(total_epochs * p2_ratio)))
            p3_epochs = max(1, int(round(total_epochs * p3_ratio)))
            while (p1_epochs + p2_epochs + p3_epochs) >= total_epochs:
                if p2_epochs > 1:
                    p2_epochs -= 1
                elif p1_epochs > 1:
                    p1_epochs -= 1
                elif p3_epochs > 1:
                    p3_epochs -= 1
                else:
                    break
            self.p1_end = p1_epochs
            self.p2_end = self.p1_end + p2_epochs
            self.p3_end = self.p2_end + p3_epochs
        else:
            self.p1_end = max(1, int(total_epochs * p1_ratio))
            self.p2_end = self.p1_end + max(1, int(total_epochs * p2_ratio))
            self.p3_end = self.p2_end + max(1, int(total_epochs * p3_ratio))
        
    def get_phase(self, epoch):
        if epoch < self.p1_end:
            return 'phase1', "Extraction de Caractéristiques (FE)"
        elif epoch < self.p2_end:
            return 'phase2a', "Renforcement de Robustesse (MATK + BFP)"
        elif epoch < self.p3_end:
            return 'phase3', "Alignement Entraînement-Test (TTA)"
        else:
            return 'phase4', "Adaptation des Têtes de Tâches"
