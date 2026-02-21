import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def aplicar_smote(X, y, sampling_strategy='auto', random_state=42, k_neighbors=5):
    """
    Aplica SMOTE para generar muestras sintéticas de la clase minoritaria.
    
    Parámetros:
    - X: DataFrame o array de características (n_samples, n_features)
    - y: array de etiquetas de clase
    - sampling_strategy: 'auto' (balancea todas las clases a la mayoría), 
                         o diccionario con número deseado por clase.
    - random_state: semilla para reproducibilidad.
    - k_neighbors: número de vecinos más cercanos a usar para la interpolación.
    
    Retorna:
    - X_resampled, y_resampled: datos balanceados.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, 
                  random_state=random_state, 
                  k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def plot_class_distribution(y_before, y_after, title_before='Antes SMOTE', title_after='Después SMOTE'):
    """
    Muestra gráficos de barras comparando la distribución de clases antes y después.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Antes
    counts_before = Counter(y_before)
    axes[0].bar(counts_before.keys(), counts_before.values(), color='skyblue')
    axes[0].set_title(title_before)
    axes[0].set_xlabel('Clase')
    axes[0].set_ylabel('Frecuencia')
    
    # Después
    counts_after = Counter(y_after)
    axes[1].bar(counts_after.keys(), counts_after.values(), color='salmon')
    axes[1].set_title(title_after)
    axes[1].set_xlabel('Clase')
    axes[1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()

