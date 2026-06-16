import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_all_classes(df: pd.DataFrame) -> pd.DataFrame:
    labels = [0, 1, 2, 4]
    display_labels = ['Negativ', 'Positiv', 'Lätt avvikande', 'Oligoklonal']

    print(classification_report(df['label'], df['final_prediction'], labels=labels, target_names=display_labels))
    cm = confusion_matrix(df['label'], df['final_prediction'], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Konfusionsmatris')
    plt.tight_layout()
    plt.show()

    if 'free_light_chain_flag' in df.columns:
        new_positives = ((df['prediction'] == 0) & (df['free_light_chain_flag'] == 1)).sum()
        new_negatives = ((df['prediction'] == 1) & (df['free_light_chain_flag'] == 0)).sum()
        changed = new_positives + new_negatives
        print(f"Antalet fall där fria lätta ändrar bedömningen: {changed}")
        print(f"Nya positiva till följd av fria lätta kedjor: {new_positives}")
        print(f"Nya negativa till följd av fria lätta kedjor: {new_negatives}")


   
    fp = sum((df['label'] == 0) & (df['final_prediction'].isin([1, 2, 4])))
    fn = sum((df['label'] == 1) & (df['final_prediction'] == 0))
    tn = sum((df['label'] == 0) & (df['final_prediction'] == 0))
    tp = sum((df['label'] == 1) & (df['final_prediction'].isin([1, 2, 4])))


    fn_rate = fn / sum(df['label'].isin([1,2,4]))
    fp_rate = fp / sum(df['label'] == 0)
    accuracy = sum(df['final_prediction'] == df['label']) / len(df['final_prediction'])
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"Accuracy:  {100*accuracy:.2f}%")
    print(f"FN-rate:   {100*fn_rate:.2f}%  (farliga missade fall)")
    print(f"FP-rate:   {100*fp_rate:.2f}%  (onödiga larm)")
    print(f"Sensitivitet:   {100*sensitivity:.2f}%  (Sannolikhet att upptäcka ett positivt fall)")
    print(f"Specificitet:   {100*specificity:.2f}%  (Sannolikhet att korrekt klassificera ett negativt fall negativt)")

    return df

def evaluate(df: pd.DataFrame) -> pd.DataFrame:

    all_probs  = np.array(df['cnn_probability'])
    all_labels = np.array(df['label'])
    if 'final_prediction' in df.columns:
        final_preds = np.array(df['final_prediction'])
    else:
        final_preds = np.array(df['prediction'])

    print(classification_report(all_labels, final_preds, target_names=['Negativ', 'Positiv']))
    print(confusion_matrix(all_labels, final_preds))

    if 'free_light_chain_flag' in df.columns:
        new_positives = ((df['prediction'] == 0) & (df['free_light_chain_flag'] == 1)).sum()
        new_negatives = ((df['prediction'] == 1) & (df['free_light_chain_flag'] == 0)).sum()
        changed = new_positives + new_negatives
        print(f"Antalet fall där fria lätta ändrar bedömningen: {changed}")
        print(f"Nya positiva till följd av fria lätta kedjor: {new_positives}")
        print(f"Nya negativa till följd av fria lätta kedjor: {new_negatives}")


   
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
