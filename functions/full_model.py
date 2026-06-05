import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from functions.analyte import ANALYTES, get_ref
from networks.cnn_network import CNNModel
import networks.cnn_network as CNN
import networks.auto_encoder as AE
from networks.auto_encoder import AutoencoderModel
from networks.inflammation_network import InflammationModel
from networks.oligoclonal_network import OligoclonalModel
from networks.inflammation_network import comment_inflammation
from functions.other_proteins import comment_albumin
from functions.other_proteins import comment_haptoglobin
from functions.other_proteins import comment_immunglobulin
from functions.free_light_chains import predict_using_free_light_chains
from functions.free_light_chains import comment_free_light_chains


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

def predict(df: pd.DataFrame, cnn_suffix: str, ae_suffix: str) -> pd.DataFrame:

    if len(df) == 0:
        raise Exception("Raden saknar nödvändig information.")
    predictor_function = get_predict_fn(cnn_suffix,ae_suffix)
    df = predictor_function(df)
    df = OligoclonalModel().predict(df)
    df = InflammationModel().predict(df)

    meta_model = joblib.load('../models/meta_model.pkl')
    df['joint_prob'] = meta_model.predict(
        np.column_stack([df['cnn_probability'], df['encoder_probability']])
    )
    df['prediction'] = (
        (df['cnn_probability'] > 0.2) | (df['proportion_gamma_region'] > 40)
    ).astype(int)

    df['alarming_free_light_chains'] = df['s_kl_kvot'].apply(
    lambda x: predict_using_free_light_chains(x) if pd.notna(x) else 0
    )

    return df

    
# ── interpret ─────────────────────────────────────────────────────────────────

def interpret(row: dict) -> dict:
    row = pd.DataFrame([row])

    interpretation = ""
    if row['final_prediction'][0] == 1:
        interpretation = "Misstänkt M-komponent. Immunfixation rekommenderas. "
    else:
        if row['cnn_probability'][0] > 0.1 and row['cnn_probability'][0] < 0.2 and row['oligoclonal_probability'][0] < 0.75:
            interpretation += "Lätt avvikande immunglobulinfördelning. M-komponent <1 g/L? Specifik immunisering? "
        elif row['oligoclonal_probability'][0] >= 0.75:
            interpretation += f"Tecken på oligoklonal fördelning. P(oligoklonalt) = {100*row['oligoclonal_probability'][0]:.1f} %. "
        else:
            interpretation += "Ingen M-komponent påvisas i serum. "

    interpretation += comment_albumin(row)
    interpretation += comment_inflammation(row)
    interpretation += comment_haptoglobin(row)
    interpretation += comment_immunglobulin(row)
    interpretation += comment_free_light_chains(row)

   

    idx = 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [3, 1]})
    ax1, ax2, ax_table = axes[0, 0], axes[1, 0], axes[0, 1]
    axes[1, 1].axis('off')



    plt.suptitle(
        f"Anomali-detektion (Klass {row.loc[idx,'label']}), id: {row.loc[idx,'id']}. "
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
    display_analytes = core_analytes + extra_analytes
    table_data, row_colors = [], []
    

    for analyte in display_analytes:
        value      = row[analyte.col].iloc[0]
        low, high  = get_ref(analyte, gender)
        outside    = not (low <= value <= high)
        flag       = " *" if outside else ""
        table_data.append([f"{value:.1f}{flag}", f"{low}–{high}"])
        row_colors.append(["#ffcccc", "#ffcccc"] if outside else ['white', 'white'])

    table = ax_table.table(
        cellText=table_data,
        rowLabels=[a.name for a in display_analytes],
        colLabels=['Värde', 'Ref'],
        loc='center', cellLoc='center'
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
        "Maskintolkning: " + interpretation +
        "\n\nLäkares tolkning: " + row.loc[idx, 'interpretation'] +
        ". TSE gamma: " + str(row.loc[idx, 'total_squared_error_gamma_region'])
    )
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


def get_predict_fn(cnn_suffix: str = None, ae_suffix: str = None):
    """
    Returnerar en predict-funktion som använder modeller med givet suffix.
    Om suffix är None används standardmodellerna.
    """
    cnn_suffix = f'_{cnn_suffix}' if cnn_suffix else ''
    ae_suffix = f'_{ae_suffix}' if ae_suffix else ''

    cnn = CNNModel(
        model_path=f'../models/convolution_model{cnn_suffix}.pth',
        scaler_path='../models/scaler.pkl'
    )
    ae = AutoencoderModel(
        model_path=f'../models/auto_encoder_model{ae_suffix}.pth'
    )

    def predict_with_models(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            raise Exception("Raden saknar nödvändig information.")
        df = cnn.predict(df)
        df = ae.predict(df)
        df = OligoclonalModel().predict(df)
        df = InflammationModel().predict(df)
        meta_model = joblib.load('../models/meta_model.pkl')
        df['joint_prob'] = meta_model.predict(
            np.column_stack([df['cnn_probability'], df['encoder_probability']])
        )
        df['prediction'] = (
            (df['cnn_probability'] > 0.2) | (df['proportion_gamma_region'] > 40)
        ).astype(int)
        return df

    return predict_with_models

