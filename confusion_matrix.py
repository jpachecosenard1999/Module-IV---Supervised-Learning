import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

def calculate_confusion_matrix(y_true, y_pred, positive_class='-', negative_class='+'):
    # Initialize counters
    TP = TN = FP = FN = 0

    # Calculate TP, TN, FP, FN values
    for true, pred in zip(y_true, y_pred):
        if true == positive_class and pred == positive_class:
            TP += 1  # True Positive
        elif true == negative_class and pred == negative_class:
            TN += 1  # True Negative
        elif true == positive_class and pred == negative_class:
            FN += 1  # False Negative
        elif true == negative_class and pred == positive_class:
            FP += 1  # False Positive

    return TP, TN, FP, FN

def plot_confusion_matrix(TP, TN, FP, FN):
    # Create the confusion matrix as a numpy array
    confusion_matrix = np.array([[TP, FN],
                                  [FP, TN]])
    
    # Create the confusion matrix plot
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='Blues')
    
    # Add a color bar
    plt.colorbar(cax)

    # Add labels for each cell in the confusion matrix
    for (i, j), value in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f"{value}", ha='center', va='center', color="black", fontsize=14)

    # Configure axis labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels(['Actual Positive', 'Actual Negative'])
    
    # Axis titles
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Show the plot
    plt.show()
    
def calculate_metrics(TP, TN, FP, FN):
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Error rate
    error_rate = (FP + FN) / (TP + TN + FP + FN)
    
    # Sensitivity (Recall or True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else None
    
    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else None
    
    # Balanced Accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Precision (Positive Predictive Value)
    precision = TP / (TP + FP) if (TP + FP) != 0 else None
    
    # MCC (Matthews Correlation Coefficient)
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else None
    
    # F1 Score
    f1_precision = 0 if (precision is None) else precision
    f1_sensitivity = 0 if (sensitivity is None) else sensitivity
    
    f1_score = (2 * precision * sensitivity) / (f1_precision + f1_sensitivity) if (f1_precision + f1_sensitivity) != 0 else None
    
    # Return all metrics
    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "mcc": mcc,
        "f1_score": f1_score
    }

def calculate_imbalance_ratio(y_test):
    # Count the frequency of each class in y_test
    classes, class_counts = np.unique(y_test, return_counts=True)
    
    # Find the majority and minority classes
    max_class = np.max(class_counts)
    min_class = np.min(class_counts)
    
    # Calculate the imbalance ratio
    imbalance_ratio = max_class / min_class
    
    return imbalance_ratio, dict(zip(classes, class_counts))

def plot_roc_curve(y_true, y_scores, positive_class='+', title='ROC Curve'):
    """
    Calculate and plot the ROC curve for binary classification.
    
    Parameters:
    - y_true: array-like, true binary labels (should match positive_class/negative_class)
    - y_scores: array-like, predicted probabilities for the positive class
    - positive_class: the label considered as positive (default '+')
    - title: title for the plot
    
    Returns:
    - fpr: false positive rates
    - tpr: true positive rates
    - roc_auc: AUC score
    """
    # Ensure y_true is binary (0/1) based on positive_class
    y_true_bin = [1 if label == positive_class else 0 for label in y_true]
    
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    return fpr, tpr, roc_auc
