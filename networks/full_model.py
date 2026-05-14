from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from analyte import ANALYTES
from analyte import get_ref



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
_models = None

hapto_globin_model = joblib.load('haptoglobin.pkl')
hapto_globin_scaler = joblib.load('haptoglobin_scaler.pkl')

def get_models():
    global _models
    if _models is None:
        print("Laddar modeller...")
        _models = load_models(device)
    return _models

def load_models(device):
    # Define model
    class CNNNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(in_channels=1,out_channels=32,kernel_size=7,padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(32,64,kernel_size=5,padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64,128,kernel_size=3,padding=1),
                nn.ReLU()
            )

            self.gap = nn.AdaptiveAvgPool1d(1)

            self.classifier = nn.Sequential(
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64,2)
            )

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.features(x)
            x = self.gap(x)
            x = torch.flatten(x,1)
            x = self.classifier(x)
            return x

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # ENCODER
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3), # 300 -> 150
                nn.ReLU(),
                nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2), # 150 -> 75
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), # 75 -> 38
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 38, 16) # Bottleneck
            )
            
            # DECODER
            self.decoder = nn.Sequential(
                nn.Linear(16, 32 * 38),
                nn.ReLU(),
                nn.Unflatten(1, (32, 38)),
                nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=0), # 38 -> 75
                nn.ReLU(),
                nn.ConvTranspose1d(16, 8, 5, stride=2, padding=2, output_padding=1), # 75 -> 150
                nn.ReLU(),
                nn.ConvTranspose1d(8, 1, 7, stride=2, padding=3, output_padding=1),  # 150 -> 300
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            z = self.encoder(x)
            out = self.decoder(z)
            return out.squeeze(1)
    
    # Define model
    class Oligoclonal(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(in_channels=1,out_channels=32,kernel_size=7,padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(32,64,kernel_size=5,padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64,128,kernel_size=3,padding=1),
                nn.ReLU()
            )

            self.gap = nn.AdaptiveAvgPool1d(1)

            self.classifier = nn.Sequential(
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64,2)
            )

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.features(x)
            x = self.gap(x)
            x = torch.flatten(x,1)
            x = self.classifier(x)
            return x
    
    class InflammationNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(300, 256),
                nn.ReLU(),
                #nn.Dropout(0.3),
                nn.Linear(256, 256),
                nn.ReLU(),
                #nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                #nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 8)
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits


    CNNModel = CNNNetwork().to(device)
    AutoEncoderModel = Autoencoder().to(device)
    OligoclonalModel = Oligoclonal().to(device)
    InflammationModel = InflammationNetwork().to(device) 
    CNNModel.load_state_dict(torch.load("convolution_model.pth", weights_only=True))
    AutoEncoderModel.load_state_dict(torch.load("auto_encoder_model.pth", weights_only=True))
    OligoclonalModel.load_state_dict(torch.load("oligoclonal_model.pth", weights_only=True))
    InflammationModel.load_state_dict(torch.load("inflammation_model.pth", weights_only=True))

    return (AutoEncoderModel,CNNModel,OligoclonalModel,InflammationModel)

def get_autoencoder_reconstructions(dataloader, AutoEncoderModel, device):
    AutoEncoderModel.eval()
    reconstructions = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            inputs = inputs[:,0:300]
            inputs = inputs.to(device)
            outputs = AutoEncoderModel(inputs)
            reconstructions.append(outputs.cpu().numpy())
            
    # Slå ihop alla batchar till en enda matris (N, 300)
    return np.concatenate(reconstructions, axis=0).squeeze()

#hitta prediktionerna från andra modellen
def predict_cnn(dataloader, CNNModel, device):
    CNNModel.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = CNNModel(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1]  # sannolikhet för positivt
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs)

#kolla oligoklonalt?
def predict_oligoclonal(dataloader, OligoclonalModel, device):
    OligoclonalModel.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = OligoclonalModel(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1]  # sannolikhet för positivt
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs)

def predict_inflammation(dataloader, InflammationModel, device) -> int:
    InflammationModel.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            inputs = inputs[:,0:300].to(device)
            logits = InflammationModel(inputs)
            probs = coral_predict_soft(logits) 
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs).astype(int)

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


def coral_predict_soft(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return probs.sum(dim=1)

def comment_albumin(albumin: float) -> str:
    if albumin < 24:
        return "Grav hypoalbuminemi. "
    if albumin < 30:
        return "Påtaglig hypoalbuminemi. "
    if albumin < 34:
        return "Hypoalbuminemi. "
    return ""

def comment_immunglobulin(row) -> str:
    comment = ""
    gender = row['gender'].iloc[0]
    
    igg_analyte = next(a for a in ANALYTES if a.col == 'igg')
    iga_analyte = next(a for a in ANALYTES if a.col == 'iga')
    igm_analyte = next(a for a in ANALYTES if a.col == 'igm')
    
    igg = row['igg'].iloc[0]
    iga = row['iga'].iloc[0]
    igm = row['igm'].iloc[0]
    
    igg_ref = get_ref(igg_analyte, gender)
    iga_ref = get_ref(iga_analyte, gender)
    igm_ref = get_ref(igm_analyte, gender)
    
    igg_normal = igg_ref[0] <= igg <= igg_ref[1]
    iga_normal = iga_ref[0] <= iga <= iga_ref[1]
    igm_normal = igm_ref[0] <= igm <= igm_ref[1]

    if igg_normal and igm_normal and iga <= 0.07:
        return "IgA-halt <0.07 g/L inger misstanke om medfödd selektiv IgA-brist. "

    for analyte, value, normal in [
        (igg_analyte, igg, igg_normal),
        (iga_analyte, iga, iga_normal),
        (igm_analyte, igm, igm_normal)
    ]:
        ref = get_ref(analyte, gender)
        if value < ref[0]:
            comment += f"Sänkt halt av {analyte.name}. "
        if value > ref[1]:
            comment += f"Förhöjd halt av {analyte.name}. "

    if igg_normal and iga_normal and igm_normal:
        comment += "Immunglobuliner med normala halter. "
    
    return comment

def comment_haptoglobin(row: pd.DataFrame) -> str:
    haptoglobin = row['haptoglobin'].iloc[0]
    age = row['age'].iloc[0]
    if haptoglobin < 0.24 and age <= 13:
        return "Sänkt haptoglobinhalt vilket kan ses vid såväl hemolys som leverpåverkan. "
    
    protein_cols = [a.col for a in ANALYTES[:5]]
    protein_value = np.array(row[protein_cols].iloc[0]).reshape(1, -1)
    protein_value = hapto_globin_scaler.transform(protein_value)
    prediction = hapto_globin_model.predict(protein_value)
    if prediction == 1:
        return "Konstellation av akutfasreaktanter förenlig med leverpåverkan och eller ökad erytrocytomsättning/hemolys. "
    return ""

    
def interpret(row: dict) -> dict:
    row = pd.DataFrame([row])
    row = predict(row['id'].tolist(),row)

    interpretation = ""
    if row['prediction'][0] == 1:
        interpretation = "Misstänkt M-komponent. Immunfixation rekommenderas. "
    
    if row['prediction'][0] == 0:
        if row['cnn_probability'][0] > 0.1 and row['cnn_probability'][0] < 0.2 and row['oligoclonal_probability'][0] < 0.75:
            interpretation += "Lätt avvikande immunglobulinfördelning. M-komponent <1 g/L ? Specifik immunisering? "
    
        elif row['oligoclonal_probability'][0] >= 0.75:
            interpretation += f"Tecken på oligoklonal fördelning. P(oligoklonalt) = {100*row['oligoclonal_probability'][0]:.1f} %. "
        else:
            interpretation += f"Ingen M-komponent påvisas i serum. "
        
    interpretation += comment_albumin(row['albumin'][0])
    interpretation += comment_inflammation(row['inflammation'][0])
    interpretation += comment_haptoglobin(row)
    interpretation += comment_immunglobulin(row)

    

    idx = 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), 
                          gridspec_kw={'height_ratios': [2, 1], 
                                       'width_ratios': [3, 1]})
    ax1 = axes[0, 0]
    ax2 = axes[1, 0]
    ax_table = axes[0, 1]  # höger kolumn för tabell
    axes[1, 1].axis('off')  # göm den nedre högra rutan

    plt.suptitle(f"Anomali-detektion (Klass {row.loc[idx,'label']}), id: {row.loc[idx,'id']}. "
                f"P(cnn)={row.loc[idx,'cnn_probability']*100:.2f}%, "
                f"P(encoder)={row.loc[idx,'encoder_probability']*100:.2f}%", 
                fontsize=11)
    plt.subplots_adjust(bottom=0.3, hspace=0.3, wspace=0.3)

    # --- Kurv-plottar (samma som innan) ---
    ax1.plot(row.loc[idx,'reconstructions'], label='Reconstruction', linewidth=1.5)
    ax1.plot(row.loc[idx,'value'], label='Original', linewidth=1.5)
    ax1.fill_between(range(300), row.loc[idx,'value'], row.loc[idx,'reconstructions'], 
                    color='red', alpha=0.2, label='Error')
    ax1.legend()

    error = row.loc[idx,'value'] - row.loc[idx,'reconstructions']
    ax2.plot(error, color='red', label='Reconstruction Error')
    ax2.fill_between(range(300), error, color='red', alpha=0.3)
    ax2.set_ylabel("Error")
    ax2.set_xlabel("Bin (Position)")
    ax2.legend()
    ax2.set_ylim((-200, 200))

    # --- Proteintabell ---
    gender = row['gender'].iloc[0]
    core_analytes = ANALYTES[:8]

    table_data = []
    row_colors = []

    for analyte in core_analytes:
        value = row[analyte.col].iloc[0]
        low, high = get_ref(analyte, gender)
        outside = not (low <= value <= high)
        flag = " *" if outside else ""
        table_data.append([f"{value:.1f}{flag}", f"{low}–{high}"])
        row_colors.append(["#ffcccc", "#ffcccc"] if outside else ['white', 'white'])

    table = ax_table.table(
        cellText=table_data,
        rowLabels=[a.name for a in core_analytes],
        colLabels=['Värde', 'Ref'],
        loc='center',
        cellLoc='center'
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

    # --- Tolkningstext ---
    tolkning_text = ("Maskintolkning: " + interpretation + 
                    "\n\nLäkares tolkning: " + row.loc[idx,'interpretation'] + 
                    ". TSE gamma: " + str(row.loc[idx,'total_squared_error_gamma_region']))

    plt.figtext(0.5, 0.20, tolkning_text,
                wrap=True,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.show()

    
    return row.iloc[0].to_dict()  # returnera som dict

def predict(ids: list[int], df: pd.DataFrame) -> pd.DataFrame:
    rows = df[ df['id'].isin([int(i) for i in ids]) ]
    protein_cols = [a.col for a in ANALYTES[:8]]  # bara de 8 första

    if len(rows) == 0:
        raise Exception("This row did not contain all neccesary information")

    X_curve     = np.array(rows['value'].tolist(),         dtype=np.float32)  # (n, 300)
    X_fractions = np.array(rows['fractions'].tolist(),  dtype=np.float32)  # (n, 6)
    X_boundaries  = np.array(rows['boundaries'].tolist(), dtype=np.float32)  # (n, 12)
    X_protein   = np.array(rows[protein_cols].values,  dtype=np.float32)  # (n, 8)


    X = np.concatenate([X_curve, X_fractions, X_boundaries,X_protein], axis=1)  # (n, 329)
    y = np.array(rows['label'].values, dtype=np.int64)

    scaler = joblib.load('scaler.pkl')
    X[:, 300:] = scaler.transform(X[:, 300:])

    X = torch.tensor(X)

    batch_sz = 512
    
    dataloader = DataLoader(
        TensorDataset(X, y, torch.tensor(np.array(ids))),
        batch_size=batch_sz,
        shuffle=False
    )

    (auto_encoder_model, cnn_model,oligoclonal_model,inflammation_model) = get_models()
    reconstructions = get_autoencoder_reconstructions(dataloader,auto_encoder_model,device)
    rows['cnn_probability'] = predict_cnn(dataloader,cnn_model,device)
    rows['oligoclonal_probability'] = predict_oligoclonal(dataloader,oligoclonal_model,device)
    rows['inflammation'] = predict_inflammation(dataloader,inflammation_model,device)

    reconstructions = reconstructions.reshape(len(rows),-1)    
    rows['reconstructions'] = list(reconstructions)

    X_curves = np.stack(rows['value'])
    squared_errors = np.power((X_curves-reconstructions),2)

    rows['total_squared_error_gamma_region'] = np.sum(squared_errors[:, 190:285], axis=1) / 1000
    rows['proportion_gamma_region'] = 100*np.sum(squared_errors[:, 190:285], axis=1) /np.sum(squared_errors[:,:],axis=1)

    logistic_model = joblib.load('logistic_regression.pkl')
    log_probabilities = logistic_model.predict_proba(np.array(rows['total_squared_error_gamma_region']).reshape(-1,1))
    rows['encoder_probability'] = log_probabilities[:,1]

    rows['prob_product'] = np.multiply(rows['cnn_probability'],rows['encoder_probability'])
    rows['prob_sum'] = np.add(rows['cnn_probability'],rows['encoder_probability'])

    meta_model = joblib.load('meta_model.pkl')
    rows['joint_prob'] = meta_model.predict(np.column_stack([rows['cnn_probability'], rows['encoder_probability']]))
    
    rows['prediction'] = ((rows['cnn_probability'] > 0.2) | (rows['proportion_gamma_region'] > 40)).astype(int)
    return rows






