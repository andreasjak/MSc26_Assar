import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from functions.analyte import ANALYTES, get_ref
from networks.cnn_dilated_convolutions import CNNModel
import networks.cnn_network as CNN
import networks.auto_encoder as AE
from networks.auto_encoder import AutoencoderModel
from networks.inflammation_network import InflammationModel
from networks.oligoclonal_network import OligoclonalModel
from networks.inflammation_network import comment_inflammation
from networks.comment_108 import Comment108Model
from functions.other_proteins import comment_albumin
from functions.other_proteins import comment_haptoglobin
from functions.other_proteins import comment_immunglobulin
from functions.free_light_chains import predict_using_free_light_chains
from functions.free_light_chains import comment_free_light_chains
from functions.urine import comment_urin
from functions.other_proteins import comment_antitrypsin
from functions.other_proteins import comment_add_urine


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Laddas en gång vid import
_cnn_model         = None
_autoencoder_model = None


def get_cnn_model() -> CNNModel:
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = CNNModel()
    return _cnn_model


def get_autoencoder_model() -> AutoencoderModel:
    global _autoencoder_model
    if _autoencoder_model is None:
        _autoencoder_model = AutoencoderModel()
    return _autoencoder_model




# ── predict ──────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame,threshold=0.2, proportion = 70) -> pd.DataFrame:

    if len(df) == 0:
        raise Exception("Raden saknar nödvändig information.")
    

    predictor_function = get_predict_fn() # här händer allt

    df = predictor_function(df)
    meta_model = joblib.load('../models/meta_model.pkl')
    df['joint_prob'] = meta_model.predict(
        np.column_stack([df['cnn_probability'], df['encoder_probability']])
    )
    df['prediction'] = (
        (df['cnn_probability'] > threshold) | (df['proportion_gamma_region'] > proportion)
    ).astype(int)

    df = predict_using_free_light_chains(df)
    df['final_prediction'] = df['prediction'].copy()
    if 'free_light_chain_flag' in df.columns:
        df.loc[df['free_light_chain_flag'] == 1, 'final_prediction'] = 1
        df.loc[df['free_light_chain_flag'] == 0, 'final_prediction'] = 0

    mask_uncertain = (df['final_prediction'] == 0) & (
    (df['comment_108'] >= 0.5) | (df['oligoclonal_probability'] >= 0.5)
    )

    df.loc[mask_uncertain & (df['comment_108'] + 0.1 > df['oligoclonal_probability']), 'final_prediction'] = 2
    df.loc[mask_uncertain & (df['comment_108'] + 0.1 <= df['oligoclonal_probability']), 'final_prediction'] = 4
    
    return df

    
# ── interpret ─────────────────────────────────────────────────────────────────

def interpret(row: dict) -> dict:
    row = pd.DataFrame([row])
    row["orders"] = [[]] 
    row["orders"] = row["orders"].astype(object)

    interpretation = ""
    interpretation += comment_albumin(row)
    interpretation += comment_inflammation(row)
    interpretation += comment_haptoglobin(row)
    interpretation += comment_antitrypsin(row)
    interpretation += comment_immunglobulin(row)
    interpretation += comment_m_component(row)
    interpretation += comment_free_light_chains(row)
    interpretation += comment_add_urine(row)
    interpretation += comment_urin(row)

    

    print(f"P(lätt avvikande)={100*row['comment_108'][0]:.2f}%")
    print(f"P(oligoklonalt)={100*row['oligoclonal_probability'][0]:.2f}%")

    idx = 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [3, 1]})
    ax1 = axes[0, 0]
    ax2 = axes[1, 0]
    ax_table = axes[0, 1]
    ax_orders = axes[1, 1]



    plt.suptitle(
        f"Anomali-detektion (Klass {row.loc[idx,'label']}), id: {row.loc[idx,'row_id']}. "
        f"P(cnn)={row.loc[idx,'cnn_probability']*100:.2f}%, "
        f"P(encoder)={row.loc[idx,'encoder_probability']*100:.2f}%",
        fontsize=11
    )
    plt.subplots_adjust(bottom=0.3, hspace=0.3, wspace=0.3)

    #ax1.plot(row.loc[idx, 'reconstructions'], label='Reconstruction', linewidth=1.5)
    ax1.plot(row.loc[idx, 'value'], label='Original', linewidth=1.5)
    #ax1.fill_between(range(300), row.loc[idx, 'value'], row.loc[idx, 'reconstructions'],
    #                 color='red', alpha=0.2, label='Error')
    ax1.legend()

    error = row.loc[idx, 'value'] - row.loc[idx, 'reconstructions']
    ax2.plot(error, color='red', label='Reconstruction Error')
    ax2.fill_between(range(300), error, color='red', alpha=0.3)
    ax2.set_ylabel("Error")
    ax2.set_xlabel("Bin (Position)")
    ax2.legend()
    ax2.set_ylim((-200, 200))

    gender        = row['gender'].iloc[0]
    core_analytes = ANALYTES[:8]
    extra_analytes = [a for a in ANALYTES[8:11] if row[a.col].iloc[0] is not None and pd.notna(row[a.col].iloc[0])]
    urine_analytes = [a for a in ANALYTES[11:] if row[a.col].iloc[0] is not None and pd.notna(row[a.col].iloc[0])]

    display_analytes = core_analytes + extra_analytes + urine_analytes
    table_data, row_colors = [], []
    

    for analyte in display_analytes:
        value      = row[analyte.col].iloc[0]
        low, high  = get_ref(analyte, gender)
        outside    = not (low <= value <= high)
        flag       = " *" if outside else ""
        if analyte.name.lower() in str(row['protein_comments']).lower() and value == 0:
            table_data.append(["Ej mätbar", f"{low}–{high}"])
            row_colors.append(["#e5ff00", "#e5ff00"])
        else:
            table_data.append([f"{value:.2f}{flag}", f"{low}–{high}"])        
            row_colors.append(["#ffcccc", "#ffcccc"] if outside else ['white', 'white'])

    ax_table.axis('off')
    ax_table.set_title('Proteiner', fontsize=10, pad=8)

    orders = row.loc[idx, "orders"]

    ax_orders.axis('off')
    ax_orders.set_title("Beställda tilläggsanalyser", fontsize=10, pad=8)

    ORDER_NAMES = {
        "1712": "Urinimmunfix",
        "0085": "Serumimmunfix",
        "1713": "Urinimmunfix (konc)"
    }
    cellText = [
    [code, ORDER_NAMES.get(code, "Okänd analys")]
    for code in orders
        ]
    

    if len(orders) > 0:
        order_table = ax_orders.table(
        cellText=cellText,
        colLabels=["Kod", "Analys"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1]
        )

        order_table.auto_set_font_size(False)
        order_table.set_fontsize(10)
        order_table.scale(1, 1.4)

        for (r, c), cell in order_table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#dddddd")
    else:
        ax_orders.text(
            0.5, 0.5,
            "Inga tilläggsbeställningar",
            ha="center",
            va="center"
        )

    # Centrera tabellen vertikalt
    table = ax_table.table(
        cellText=table_data,
        rowLabels=[a.name for a in display_analytes],
        colLabels=['Värde', 'Ref'],
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]  # [x, y, width, height] – fyller hela axeln
    )
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor('#dddddd')
        elif row_idx > 0 and col_idx >= 0:
            cell.set_facecolor(row_colors[row_idx - 1][col_idx])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax_table.axis('off')
    ax_table.set_title('Proteiner', fontsize=10, pad=8)

    tolkning_text = (
        "Maskintolkning:\n" + interpretation +
        "\n\nLäkares tolkning:\n" + row.loc[idx, 'interpretation']  )
    plt.figtext(0.5, 0.20, tolkning_text, wrap=True,
                horizontalalignment='center', verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.show()

    return row.iloc[0].to_dict()

def retrain(train_rows, val_rows, model_suffix: str):
    cnn_path = f'../models/convolution_model_{model_suffix}.pth'
    ae_path  = f'../models/auto_encoder_model_{model_suffix}.pth'
    scaler_path = f'../models/scaler.pkl'

    cnn = CNNModel(model_path=cnn_path, scaler_path=scaler_path)
    ae = AutoencoderModel(model_path=ae_path)
    cnn_train_dl, cnn_val_dl, _ = CNN.build_dataloaders(train_rows, val_rows, val_rows)
    ae_train_dl,  ae_val_dl,  _ = AE.build_dataloaders(train_rows, val_rows, val_rows)

    print("Beginning re-training of the CNN-model!\n")
    cnn.retrain(cnn_train_dl, cnn_val_dl, model_path=cnn_path,patience=10)

    print("Beginning re-training of the AE-model!\n")
    ae.retrain(ae_train_dl, ae_val_dl, model_path=ae_path,patience=10)

    print(f"Träning klar! Modeller sparade med suffix '{model_suffix}'")


def get_predict_fn():
    """
    Returnerar en predict-funktion som använder modeller med givet suffix.
    Om suffix är None används standardmodellerna.
    """

    cnn = CNNModel()
    ae = AutoencoderModel()

    def predict_with_models(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            raise Exception("Raden saknar nödvändig information.")
        df = cnn.predict(df)
        df = ae.predict(df)
        df = OligoclonalModel().predict(df)
        df = InflammationModel().predict(df)
        df = Comment108Model().predict(df)
        return df

    return predict_with_models

def comment_m_component(row):
    interpretation = ''
    if row['final_prediction'][0] == 1:
        interpretation += "Misstänkt M-komponent. "
        current_orders = row.at[0, "orders"]
        row.at[0, "orders"] = current_orders + ["0085"]
    elif max(row['comment_108'][0], row['oligoclonal_probability'][0]) >= 0.5:
        if(row['comment_108'][0] + 0.1 > row['oligoclonal_probability'][0]):
               interpretation += f"Lätt avvikande immunglobulinfördelning. M-komponent <1 g/L? Specifik immunisering? "
        else:
            interpretation += f"Oligoklonal fördelning av immunglobulinerna. "
    else:
        interpretation += "Ingen M-komponent påvisas i serum. "

    return interpretation