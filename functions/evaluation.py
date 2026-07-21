import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from functions.analyte import ANALYTES, get_ref


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

def evaluate(df: pd.DataFrame,threshold=0.5) -> pd.DataFrame:

    all_probs  = np.array(df['cnn_probability'])
    all_labels = np.array(df['label'])
    if 'prediction' not in df.columns:
        df['prediction'] = (df['cnn_probability'] >= threshold).astype(int)
    if 'final_prediction' in df.columns:
        final_preds = np.array(df['final_prediction'])
        cm = confusion_matrix(df['label'], df['final_prediction'])
    else:
        final_preds = np.array(df['prediction'])
        cm = confusion_matrix(df['label'], df['prediction'])
    print(np.unique(all_labels))
    print(np.unique(final_preds))
    display_labels = ['Negativ', 'Positiv']
    print(classification_report(all_labels, final_preds, target_names=['Negativ', 'Positiv']))
    print(confusion_matrix(all_labels, final_preds))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    _, ax = plt.subplots(figsize=(8, 6))
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

def show_case(row):
    idx = 0

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[3, 1]
    )

    ax_curve = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    fig.suptitle(
        f"ID {row.loc[idx,'row_id']}    "
        f"Label: {row.loc[idx,'label']}    "
        f"CNN={100*row.loc[idx,'cnn_probability']:.1f}%    "
        f"Oligoklonal={100*row.loc[idx,'oligoclonal_probability']:.1f}%    "
        f"Kommentar108={100*row.loc[idx,'comment_108']:.1f}%",
        fontsize=12
    )

    # Kurva
    ax_curve.plot(row.loc[idx, "value"], lw=2)
    ax_curve.set_xlabel("Position")
    ax_curve.set_ylabel("Signal")
    ax_curve.set_title("Capillary electrophoresis")

    # ---------- Proteiner ----------
    gender = row["gender"].iloc[0]

    core_analytes = ANALYTES[:8]
    extra_analytes = [
        a for a in ANALYTES[8:11]
        if pd.notna(row[a.col].iloc[0])
    ]
    urine_analytes = [
        a for a in ANALYTES[11:]
        if pd.notna(row[a.col].iloc[0])
    ]

    analytes = core_analytes + extra_analytes + urine_analytes

    table_data = []
    colors = []

    for analyte in analytes:

        value = row[analyte.col].iloc[0]
        low, high = get_ref(analyte, gender)

        outside = not (low <= value <= high)

        table_data.append([
            f"{value:.2f}",
            f"{low}-{high}"
        ])

        colors.append(
            ["#ffdddd", "#ffdddd"] if outside else ["white", "white"]
        )

    ax_table.axis("off")

    table = ax_table.table(
        cellText=table_data,
        rowLabels=[a.name for a in analytes],
        colLabels=["Value", "Reference"],
        cellLoc="center",
        bbox=[0,0,1,1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r,c), cell in table.get_celld().items():

        if r == 0:
            cell.set_facecolor("#dddddd")

        elif r > 0 and c >= 0:
            cell.set_facecolor(colors[r-1][c])

    # ---------- Läkartolkning ----------
    ax_text.axis("off")

    interpretation = str(row.loc[idx, "interpretation"])

    ax_text.text(
        0,
        1,
        "Physician interpretation\n\n" + interpretation,
        va="top",
        fontsize=11,
        wrap=True
    )

    plt.tight_layout()

    return fig

def generate_latex_table(df):
    tn = df['tn'].mean()
    tn_std = df['tn'].std()
    fp = df['fp'].mean()
    fp_std = df['fp'].std()
    fn = df['fn'].mean()
    fn_std = df['fn'].std()
    tp = df['tp'].mean()
    tp_std = df['tp'].std()
    acc = df['val_accuracy'].mean()
    acc_std = df['val_accuracy'].std()
    auc = df['val_auc'].mean()
    auc_std = df['val_auc'].std()
    sens = df['val_sens'].mean()
    sens_std = df['val_sens'].std()
    spec = df['val_spec'].mean()
    spec_std = df['val_spec'].std()
    loss = df['val_loss'].mean()
    loss_std = df['val_loss'].std()
    return r"""
\begin{table}[H]
\begin{center}
\begin{minipage}{0.45\textwidth}
    \centering
    \begin{tabular}{cc|cc}
      \toprule
      & & \multicolumn{2}{c}{Predicted} \\
      & & Neg. & Pos. \\
      \midrule
      \multirow{2}{*}{\rotatebox[origin=c]{90}{Actual}}
      & Neg. & $""" + f"{tn:.0f} \\pm {tn_std:.0f}" + r"""$ & $""" + f"{fp:.0f} \\pm {fp_std:.0f}" + r"""$ \\[0.1em]
      & Pos. & $""" + f"{fn:.0f} \\pm {fn_std:.0f}" + r"""$ & $""" + f"{tp:.0f} \\pm {tp_std:.0f}" + r"""$ \\
      \addlinespace[0.6em]
      \bottomrule
    \end{tabular}
\end{minipage}%
\begin{minipage}{0.45\textwidth}
    \centering
    \begin{tabular}{lr}
      \toprule
      Accuracy    & $""" + f"{acc:.2f} \\pm {acc_std:.2f}" + r"""$ \% \\
      AUC         & $""" + f"{auc:.4f} \\pm {auc_std:.4f}" + r"""$ \\
      Sensitivity & $""" + f"{sens:.4f} \\pm {sens_std:.4f}" + r"""$ \\
      Specificity & $""" + f"{spec:.4f} \\pm {spec_std:.4f}" + r"""$ \\
      Avg. Val. loss & $""" + f"{loss:.4f} \\pm {loss_std:.4f}" + r"""$ \\
      \bottomrule
    \end{tabular}
\end{minipage}
\end{center}
\caption{""" + "Caption"  + r"""}
\label{""" + "label" + r"""}
\end{table}
"""

def plot_3_by_3_confusion_matrix(df):
    cm = confusion_matrix(df['label'], df['final_prediction'])

    labels = [
        'No M-component',
        'M-component',
        'Oligoclonal/heterogeneity'
    ]


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )
    fig, ax = plt.subplots(figsize=(7, 7))

    disp.plot(
        ax=ax,
        
        colorbar=False,
        cmap='Blues',
        values_format='d'
    )

    for text in ax.texts:
        text.set_fontsize(16)

    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlabel('Predicted class', fontsize=15, labelpad=15)
    ax.set_ylabel('True class', fontsize=15, labelpad=15)

    ax.set_title('Confusion Matrix', fontsize=17, pad=20)

    fig.tight_layout()

    # efter tight_layout:
    fig.subplots_adjust(
        left=0.18,
        bottom=0.22
    )

    plt.show()