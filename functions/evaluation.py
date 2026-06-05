import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def evaluate(df: pd.DataFrame, threshold=0.2, proportion=60) -> pd.DataFrame:

    all_probs  = np.array(df['cnn_probability'])
    all_labels = np.array(df['label'])

    if 'proportion_gamma_region' in df.columns:
        all_preds  = ((all_probs >= threshold) | (df['proportion_gamma_region'] > proportion)).astype(int)
        print('using autoencoder model')
    else:
        all_preds = (all_probs >= threshold).astype(int)

    df['prediction'] = all_preds
    df['final_prediction'] = df['prediction'].copy()
    df.loc[df['alarming_free_light_chains'] == 1, 'final_prediction'] = 1
    final_preds = df['final_prediction']
    print(classification_report(all_labels, final_preds, target_names=['Negativ', 'Positiv']))
    print(confusion_matrix(all_labels, final_preds))

    new_positives = ((df['prediction'] == 0) & (df['alarming_free_light_chains'] == 1)).sum()
    print(f"Nya positiva till följd av fria lätta kedjor: {new_positives}")

   
    # ROC-kurva
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # AUC
    roc_auc = auc(fpr, tpr)

    print("AUC:", roc_auc)

    fp = sum((all_labels == 0) & (final_preds == 1))
    fn = sum((all_labels == 1) & (final_preds == 0))
    tn = sum((all_labels == 0) & (final_preds == 0))
    tp = sum((all_labels == 1) & (final_preds == 1))


    fn_rate = fn / sum(all_labels == 1)
    fp_rate = fp / sum(all_labels == 0)
    accuracy = sum(final_preds == all_labels) / len(final_preds)
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
