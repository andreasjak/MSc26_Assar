import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

def calculate_metrics(df: pd.DataFrame) -> dict:
        if 'cnn_probability' in df.columns:
            probs  = np.array(df['cnn_probability'])
        else:
            probs = np.array(df['probability'])
        labels = np.array(df['label'])
        preds = (probs >= 0.5).astype(int)

        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        tn = np.sum((labels == 0) & (preds == 0))
        tp = np.sum((labels == 1) & (preds == 1))

        accuracy = sum(preds == labels) / len(preds)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        roc_auc = roc_auc_score(labels, probs)

        fn_rate = fn / np.sum(labels == 1)
        fp_rate = fp / np.sum(labels == 0)



        return {
            "accuracy": accuracy,
            "auc": roc_auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "fn_rate": fn_rate,
            "fp_rate": fp_rate,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }