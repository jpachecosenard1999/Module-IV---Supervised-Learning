import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

def calculate_confusion_matrix(y_true, y_pred):
    # Obtener todas las clases presentes en y_true e y_pred
    classes = sorted(list(set(y_true) | set(y_pred)))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Mapeo de clase a índice
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Llenar la matriz de confusión
    for true, pred in zip(y_true, y_pred):
        true_idx = class_to_idx[true]
        pred_idx = class_to_idx[pred]
        cm[true_idx][pred_idx] += 1
    
    return cm, classes

def plot_confusion_matrix(confusion_matrix, classes):
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues')
    plt.colorbar(cax)
    
    # Configurar ejes
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    
    # Etiquetas
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Anotar valores
    for (i, j), value in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f"{value}", ha='center', va='center', color='black')
    
    plt.show()

def calculate_metrics(confusion_matrix, classes):
    metrics = {}
    n = len(classes)
    total = np.sum(confusion_matrix)
    
    # Métricas por clase
    class_metrics = {}
    for i in range(n):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        tn = total - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        class_metrics[classes[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(confusion_matrix[i, :])
        }
    
    # Métricas globales
    accuracy = np.trace(confusion_matrix) / total
    
    # Precisión macro/micro/weighted
    precisions = [cm['precision'] for cm in class_metrics.values()]
    recalls = [cm['recall'] for cm in class_metrics.values()]
    balanced_accuracy = np.mean(recalls)
    f1_scores = [cm['f1'] for cm in class_metrics.values()]
    supports = [cm['support'] for cm in class_metrics.values()]
    
    metrics['accuracy'] = accuracy
    metrics['balanced_accuracy'] = balanced_accuracy
    metrics['precision_macro'] = np.mean(precisions)
    metrics['recall_macro'] = np.mean(recalls)
    metrics['f1_macro'] = np.mean(f1_scores)
    metrics['precision_weighted'] = np.average(precisions, weights=supports)
    metrics['recall_weighted'] = np.average(recalls, weights=supports)
    metrics['f1_weighted'] = np.average(f1_scores, weights=supports)
    
    # MCC multiclase (fórmula corregida)
    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)
    total = np.sum(confusion_matrix)
    
    cov_xy = np.sum(confusion_matrix * confusion_matrix)
    cov_xx = np.dot(row_sums, row_sums)
    cov_yy = np.dot(col_sums, col_sums)
    
    mcc_numerator = cov_xy - (cov_xx * cov_yy) / (total ** 2)
    mcc_denominator = np.sqrt((cov_xx - cov_xx**2 / total**2) * (cov_yy - cov_yy**2 / total**2))
    
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
    
    metrics['mcc'] = mcc
    return metrics

def calculate_imbalance_ratio(y_test):
    classes, counts = np.unique(y_test, return_counts=True)
    if len(counts) < 2:
        return 0.0, dict(zip(classes, counts))
    max_count = max(counts)
    min_count = min(counts)
    return max_count / min_count, dict(zip(classes, counts))

def plot_multiclass_roc_curve(y_true, y_scores, classes=None, average='macro', title='Multiclass ROC Curve'):
    """
    Plot ROC curves for each class (one-vs-rest) and compute macro/micro AUC.
    
    Parameters:
    - y_true: array-like, true class labels
    - y_scores: array-like of shape (n_samples, n_classes), predicted probabilities per class
    - classes: list of class names in the order corresponding to columns of y_scores.
               If None, they are inferred from y_true.
    - average: 'macro' or 'micro' for overall AUC
    - title: title for the plot
    
    Returns:
    - roc_auc_macro: macro-average AUC
    - roc_auc_micro: micro-average AUC (if applicable)
    - per_class_auc: dict with AUC per class
    """
    if classes is None:
        classes = sorted(set(y_true))
    n_classes = len(classes)
    
    # Binarize the true labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=classes)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict()
    
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc_per_class[cls] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    # Compute macro-average AUC (simple mean of per-class AUCs)
    roc_auc_macro = np.mean(list(roc_auc_per_class.values()))
    
    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'Class {cls} (AUC = {roc_auc_per_class[cls]:.2f})')
    
    # Plot micro-average
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', lw=4,
             label=f'Micro-average (AUC = {roc_auc_micro:.2f})')
    
    # Plot diagonal line for random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    return roc_auc_macro, roc_auc_micro, roc_auc_per_class