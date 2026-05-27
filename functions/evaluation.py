import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def evaluate(df: pd.DataFrame, threshold=0.2, proportion=60) -> pd.DataFrame:

    all_probs  = np.array(df['cnn_probability'])
    all_labels = np.array(df['label'])
    all_ids    = np.array(df['id'])

    if 'proportion_gamma_region' in df.columns:
        all_preds  = ((all_probs >= threshold) | (df['proportion_gamma_region'] > proportion)).astype(int)
        
    else:
        all_preds = (all_probs >= threshold).astype(int)

    df['prediction'] = all_preds
    print(classification_report(all_labels, all_preds, target_names=['Negativ', 'Positiv']))
    print(confusion_matrix(all_labels, all_preds))

    # ROC-kurva
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # AUC
    roc_auc = auc(fpr, tpr)

    print("AUC:", roc_auc)

    fp = sum((all_labels == 0) & (all_preds == 1))
    fn = sum((all_labels == 1) & (all_preds == 0))
    tn = sum((all_labels == 0) & (all_preds == 0))
    tp = sum((all_labels == 1) & (all_preds == 1))


    fn_rate = fn / sum(all_labels == 1)
    fp_rate = fp / sum(all_labels == 0)
    accuracy = sum(all_preds == all_labels) / len(all_preds)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"Accuracy:  {100*accuracy:.2f}%")
    print(f"FN-rate:   {100*fn_rate:.2f}%  (farliga missade fall)")
    print(f"FP-rate:   {100*fp_rate:.2f}%  (onödiga larm)")
    print(f"Sensitivitet:   {100*sensitivity:.2f}%  (Sannolikhet att upptäcka ett positivt fall)")
    print(f"Specificitet:   {100*specificity:.2f}%  (Sannolikhet att korrekt klassificera ett negativt fall negativt)")
    print(f"AUC: {roc_auc:.4f}")

    # Rita kurvan
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.show()


    return df
