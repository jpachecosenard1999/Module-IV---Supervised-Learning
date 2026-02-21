import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder

def anova_feature_selection(X, y, k=None):
    """
    Calcula el estadístico F y el p-valor de ANOVA para cada característica.
    
    Parámetros:
    - X: DataFrame o array de características (n_samples, n_features)
    - y: array de etiquetas de clase
    - k: número de características a seleccionar (si es None, se devuelven todas ordenadas)
    
    Retorna:
    - scores: array con los valores F
    - p_values: array con los p-valores
    - Si k no es None, también retorna los índices de las top k características.
    """
    # Asegurar que y sea 1D y numérico para f_classif
    if isinstance(y, pd.Series):
        y = y.values
    # Codificar etiquetas si son categóricas
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Calcular ANOVA F
    f_scores, p_vals = f_classif(X, y)
    
    # Ordenar por F-score descendente
    sorted_idx = np.argsort(f_scores)[::-1]
    
    if k is not None:
        top_idx = sorted_idx[:k]
        return f_scores, p_vals, top_idx
    else:
        return f_scores, p_vals

def intra_class_deviation(X, y):
    """
    Calcula la desviación estándar promedio dentro de cada clase para cada característica.
    Una baja desviación intra-clase indica que los valores de la característica son
    consistentes dentro de la misma clase.
    
    Parámetros:
    - X: DataFrame o array de características (n_samples, n_features)
    - y: array de etiquetas de clase
    
    Retorna:
    - dev_scores: array con la desviación intra-clase promedio por característica.
    - por clase: (opcional) matriz de desviaciones por clase (n_classes, n_features)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    clases = np.unique(y)
    n_features = X.shape[1]
    
    intra_std = np.zeros((len(clases), n_features))
    
    for i, cls in enumerate(clases):
        mask = (y == cls)
        X_cls = X[mask]
        if X_cls.shape[0] > 1:
            intra_std[i, :] = np.std(X_cls, axis=0, ddof=1)  
        else:
            intra_std[i, :] = 0  
    
    # Promedio sobre clases 
    class_counts = np.array([np.sum(y == cls) for cls in clases])
    weights = class_counts / np.sum(class_counts)
    avg_intra_std = np.average(intra_std, axis=0, weights=weights)
    
    return avg_intra_std, intra_std

def combined_ranking(X, y, alpha=1.0):
    """
    Combina el estadístico F (mayor es mejor) y la desviación intra-clase (menor es mejor)
    en un único score de ranking.
    
    Score combinado = F - alpha * intra_deviation
    (se normalizan ambas métricas para que estén en escalas comparables)
    
    Parámetros:
    - X, y: datos
    - alpha: peso de la desviación intra-clase (por defecto 1.0)
    
    Retorna:
    - DataFrame con características, F, intra_dev, score combinado y ranking.
    """
    f_scores, p_vals = anova_feature_selection(X, y)
    intra_dev, _ = intra_class_deviation(X, y)
    
    # Normalizar para que tengan media 0 y desviación 1 (z-score)
    f_norm = (f_scores - np.mean(f_scores)) / np.std(f_scores)
    intra_norm = (intra_dev - np.mean(intra_dev)) / np.std(intra_dev)
    
    combined = f_norm - alpha * intra_norm
    
    # Crear DataFrame
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'F_statistic': f_scores,
        'p_value': p_vals,
        'Intra_class_std': intra_dev,
        'Combined_score': combined
    })
    df['Rank'] = df['Combined_score'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values('Rank').reset_index(drop=True)
    return df

def plot_feature_importance(scores, feature_names=None, title='Importancia de características', top_n=None):
    """
    Gráfico de barras con las puntuaciones de las características.
    """
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(len(scores))]
    
    if top_n is not None:
        idx = np.argsort(scores)[-top_n:][::-1]
        scores = scores[idx]
        feature_names = [feature_names[i] for i in idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(scores)), scores, color='steelblue')
    plt.yticks(range(len(scores)), feature_names)
    plt.xlabel('Puntuación')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
