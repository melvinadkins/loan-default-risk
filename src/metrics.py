import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score, 
    f1_score,
    roc_auc_score
)

from scipy.stats import ks_2samp


def display_classification_metric(model, X, y, threshold=0.5, title=None, save=True):
    """
    Displays classification report and confusion matrix for a given model and dataset.

    Parameters:
    -----------
    model: Trained classification model with a predict_proba method.
    X: Feature data to predict on.
    y: True target labels.
    title: Optional title for the output.
    threshold: Classification threshold for converting probabilities to class labels.

    Returns:
    --------
    None (prints classification report and confusion matrix)
    
    """
    title = title or "Confusion Matrix"

    # Get predictions
    y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print()

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    if save:
        plt.savefig("../artifacts/" + title.replace(" ", "_").lower() + "_confusion_matrix.png")
    plt.show()

def optimize_threshold(model, X, y):
    """
    Optimizes classification threshold to maximize F1 score for a given model and dataset.

    Parameters:
    -----------
    model: Trained classification model with a predict_proba method.
    X: Feature data to predict probabilities on.
    y: True target labels.
    metric: Metric to optimize F1.

    Returns:
    --------
    optimal_threshold (float): Threshold that optimizes the F1 score.
    
    """
    # Threshold optimization
    thresholds = np.arange(0, 1.01, 0.01)  
    best_f1 = -1
    best_threshold = 0
    recall_scores, precision_scores, f1_scores = [], [], []

    # Predict probability of positive class
    y_proba = model.predict_proba(X)[:, 1]

    for threshold in thresholds:
        y_pred_thresh = (y_proba > threshold).astype(int)
        
        precision = precision_score(y, y_pred_thresh, zero_division = 0)
        recall = recall_score(y, y_pred_thresh, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        recall_scores.append(recall)
        precision_scores.append(precision)
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            print(f"Threshold={threshold:.2f} Recall={recall:.2f} Precision={precision:.2f} F1={f1:.3f}")
    print(f"Best Threshold: {best_threshold}")

    # Plot metric scores across thresholds
    plt.figure(figsize = (10, 6))
    plt.plot(thresholds, recall_scores, label = "Recall", marker = "o")
    plt.plot(thresholds, precision_scores, label = "Precision", marker = "o")
    plt.plot(thresholds, f1_scores, label = "F1 Score", marker = "o")
    plt.axvline(x = best_threshold, color = "red", linestyle = "--", label = f"Best Threshold: {best_threshold:.2f}")
    plt.title("Precision, Recall, and F1 Score Across Thresholds")
    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_threshold

def compute_ks(X, y, model):
    y_proba = model.predict_proba(X)[:, 1]
    ks_stat, p_value = ks_2samp(y_proba[y == 0], y_proba[y == 1])
    return ks_stat, p_value

def compute_performance_metrics(X, y, model, threshold):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(
        y,
        y_proba
    )

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    ks_stat, p_value = compute_ks(X, y, model)

    return {
        "Threshold": threshold,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "ks_stat": ks_stat,
        "ks_p_value": p_value
    }