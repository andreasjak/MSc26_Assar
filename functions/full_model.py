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


# ── Inflammation ────────────────────────────────────────────────────────────

class InflammationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def coral_predict_soft(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).sum(dim=1)


def predict_inflammation(df: pd.DataFrame) -> pd.DataFrame:
    model = InflammationNetwork().to(device)
    model.load_state_dict(torch.load('../models/inflammation_model.pth', weights_only=True))
    model.eval()

    X = torch.tensor(np.array(df['value'].tolist(), dtype=np.float32))
    dataloader = DataLoader(TensorDataset(X), batch_size=512)

    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0][:, 0:300].to(device)
            logits = model(inputs)
            probs = coral_predict_soft(logits)
            all_probs.append(probs.cpu().numpy())

    df['inflammation'] = np.concatenate(all_probs).astype(int)
    return df


# ── Oligoclonal ─────────────────────────────────────────────────────────────

class OligoclonalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def predict_oligoclonal(df: pd.DataFrame) -> pd.DataFrame:
    model = OligoclonalNetwork().to(device)
    model.load_state_dict(torch.load('../models/oligoclonal_model.pth', weights_only=True))
    model.eval()

    X = torch.tensor(np.array(df['value'].tolist(), dtype=np.float32))
    dataloader = DataLoader(TensorDataset(X), batch_size=512)

    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())

    df['oligoclonal_probability'] = np.concatenate(all_probs)
    return df


# ── Haptoglobin ─────────────────────────────────────────────────────────────

hapto_model  = joblib.load('../models/haptoglobin.pkl')
hapto_scaler = joblib.load('../models/haptoglobin_scaler.pkl')


def comment_haptoglobin(row: pd.DataFrame) -> str:
    haptoglobin = row['haptoglobin'].iloc[0]
    age         = row['age'].iloc[0]
    if haptoglobin < 0.24 and age <= 13:
        return "Sänkt haptoglobinhalt vilket kan ses vid såväl hemolys som leverpåverkan. "
    protein_cols  = [a.col for a in ANALYTES[:5]]
    protein_value = np.array(row[protein_cols].iloc[0]).reshape(1, -1)
    protein_value = hapto_scaler.transform(protein_value)
    prediction    = hapto_model.predict(protein_value)
    if prediction == 1:
        return "Konstellation av akutfasreaktanter förenlig med leverpåverkan och eller ökad erytrocytomsättning/hemolys. "
    return ""


# ── Kommentarsfunktioner ─────────────────────────────────────────────────────

def comment_albumin(albumin: float) -> str:
    if albumin < 24: return "Grav hypoalbuminemi. "
    if albumin < 30: return "Påtaglig hypoalbuminemi. "
    if albumin < 34: return "Hypoalbuminemi. "
    return ""


def comment_inflammation(severity: int) -> str:
    match severity:
        case 0: return "Inga tecken på inflammation. "
        case 1: return "Tecken på diskret inflammation. "
        case 2: return "Tecken på lätt inflammation. "
        case 3: return "Tecken lätt-måttlig inflammation. "
        case 4: return "Tecken på måttlig inflammation. "
        case 5: return "Tecken på måttlig-kraftig inflammation. "
        case 6: return "Tecken på kraftig inflammation. "
        case 7: return "Tecken på mycket kraftig inflammation. "


def comment_immunglobulin(row: pd.DataFrame) -> str:
    comment = ""
    gender = row['gender'].iloc[0]

    igg_a = next(a for a in ANALYTES if a.col == 'igg')
    iga_a = next(a for a in ANALYTES if a.col == 'iga')
    igm_a = next(a for a in ANALYTES if a.col == 'igm')

    igg, iga, igm = row['igg'].iloc[0], row['iga'].iloc[0], row['igm'].iloc[0]

    igg_normal = get_ref(igg_a, gender)[0] <= igg <= get_ref(igg_a, gender)[1]
    iga_normal = get_ref(iga_a, gender)[0] <= iga <= get_ref(iga_a, gender)[1]
    igm_normal = get_ref(igm_a, gender)[0] <= igm <= get_ref(igm_a, gender)[1]

    if igg_normal and igm_normal and iga <= 0.07:
        return "IgA-halt <0.07 g/L inger misstanke om medfödd selektiv IgA-brist. "

    for analyte, value in [(igg_a, igg), (iga_a, iga), (igm_a, igm)]:
        low, high = get_ref(analyte, gender)
        if value < low:  comment += f"Sänkt halt av {analyte.name}. "
        if value > high: comment += f"Förhöjd halt av {analyte.name}. "

    if igg_normal and iga_normal and igm_normal:
        comment += "Immunglobuliner med normala halter. "

    return comment


# ── predict ──────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, cnn_suffix: str, ae_suffix: str) -> pd.DataFrame:

    if len(df) == 0:
        raise Exception("Raden saknar nödvändig information.")
    predictor_function = get_predict_fn(cnn_suffix,ae_suffix)
    df = predictor_function(df)
    df = predict_oligoclonal(df)
    df = predict_inflammation(df)

    meta_model = joblib.load('../models/meta_model.pkl')
    df['joint_prob'] = meta_model.predict(
        np.column_stack([df['cnn_probability'], df['encoder_probability']])
    )
    df['prediction'] = (
        (df['cnn_probability'] > 0.2) | (df['proportion_gamma_region'] > 40)
    ).astype(int)

    return df

def predict_inflammation_and_oligoclonal(df: pd.DataFrame) -> pd.DataFrame:
    df = predict_oligoclonal(df)
    df = predict_inflammation(df)
    return df

# ── interpret ─────────────────────────────────────────────────────────────────

def interpret(row: dict) -> dict:
    row = pd.DataFrame([row])

    interpretation = ""
    if row['prediction'][0] == 1:
        interpretation = "Misstänkt M-komponent. Immunfixation rekommenderas. "
    else:
        if row['cnn_probability'][0] > 0.1 and row['cnn_probability'][0] < 0.2 and row['oligoclonal_probability'][0] < 0.75:
            interpretation += "Lätt avvikande immunglobulinfördelning. M-komponent <1 g/L? Specifik immunisering? "
        elif row['oligoclonal_probability'][0] >= 0.75:
            interpretation += f"Tecken på oligoklonal fördelning. P(oligoklonalt) = {100*row['oligoclonal_probability'][0]:.1f} %. "
        else:
            interpretation += "Ingen M-komponent påvisas i serum. "

    interpretation += comment_albumin(row['albumin'][0])
    interpretation += comment_inflammation(row['inflammation'][0])
    interpretation += comment_haptoglobin(row)
    interpretation += comment_immunglobulin(row)

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
    table_data, row_colors = [], []

    for analyte in core_analytes:
        value      = row[analyte.col].iloc[0]
        low, high  = get_ref(analyte, gender)
        outside    = not (low <= value <= high)
        flag       = " *" if outside else ""
        table_data.append([f"{value:.1f}{flag}", f"{low}–{high}"])
        row_colors.append(["#ffcccc", "#ffcccc"] if outside else ['white', 'white'])

    table = ax_table.table(
        cellText=table_data,
        rowLabels=[a.name for a in core_analytes],
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

