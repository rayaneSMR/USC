"""
Modélisation Paramétrique de la Perte Sémantique (§5).
Ajustement du modèle analytique bidimensionnel unifié de Nguyen et al.
"""
import numpy as np
from scipy.optimize import curve_fit

def _semantic_model_func(X, mu0, mu1, mu2, mu3, mu4, mu5, mu6):
    """
    Modèle paramétrique bidimensionnel unifié.
    X[0] = rho (taux de compression sémantique)
    X[1] = gamma (SNR optique reçu)
    Note: Simplifié pour N_c = 1 pour un ajustement stable.
    """
    rho, gamma = X
    logistic_term = mu2 / (1 + np.exp(-mu3 * gamma - mu4))
    exp_term = np.exp(mu5 * rho) + mu6 * rho
    return mu0 + (mu1 + logistic_term) * exp_term

def fit_semantic_surface(rhos, gammas, metric_values):
    """
    Ajuste la surface sémantique à partir de points de données discrets.
    rhos: Taux de compression (array 1D)
    gammas: SNRs optiques (array 1D)
    metric_values: Métriques de performance (ex: mIoU, Accuracy) (array 1D)
    """
    X_data = np.vstack((rhos, gammas))
    
    # Bornes initiales raisonnables pour la convergence
    p0 = [0.1, 0.1, 0.5, 1.0, 0.0, -1.0, 0.1]
    
    try:
        popt, _ = curve_fit(_semantic_model_func, X_data, metric_values, p0=p0, maxfev=10000)
        return popt
    except RuntimeError:
        print("Erreur : La descente de gradient n'a pas convergé pour le fit.")
        return None

def predict_semantic_performance(rho, gamma, popt):
    """
    Prédit la performance sémantique continue pour optimiser l'allocation de ressources.
    """
    return _semantic_model_func((rho, gamma), *popt)
