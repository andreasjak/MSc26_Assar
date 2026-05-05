import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import re
import joblib



def extract_fractions(delimit_list):
    fractions = []
    for item in delimit_list:
        match = re.search(r'(\d+\.?\d*)\s*%', str(item))
        if match:
            fractions.append(float(match.group(1)))
    return fractions

def extract_relevant_values(analysis_list: list[str], value_list: list[str], relevant: list[str]) -> np.array:
    protein_values = np.array([np.nan]*len(relevant))
    mapping = dict()

    if len(analysis_list) > len(value_list):
        raise Exception("Analysis list and value list must be of same length!")
    
    for i in range(len(analysis_list)):
        analysis = analysis_list[i].replace('*','')
        value = clean_value(value_list[i])
        mapping.update({analysis: value})

    for (i, analysis) in enumerate(relevant):
        protein_values[i] = mapping.get(analysis,np.nan)

    #får vi en nan får vi tyvärr hoppa över hela raden, jag kör pd.dropna senare
    if sum( (np.isnan(protein_values) == True) ) > 0:
        return np.nan

    return protein_values


def clean_value(val: str) -> float:
    if val is None or val == '' or val == 'NA' or val == 'NEG':
        return np.nan
    try:
        return float(str(val).replace('<', '').replace('>', '').strip())
    except:
        return np.nan
    
def pre_process_data(ids: list[any], df: pd.DataFrame) -> pd.DataFrame:

    rows = df[ df['id'].isin(ids) ] #hämta ut rätt rader

    rows = rows[ rows['value'].apply(len) == 301 ] #kasta de rader där value inte har rätt längd (typ bara en korrupt rad)
    rows['value'] = rows['value'].apply(lambda x: x[:300]) #kasta sista värdet (som alltid 0)

    #hämta ut protein-value
    relevant_anlysis = ['0054','0055','0056','0058','0062','0064','0065','0066']
    rows['protein_value'] = rows.apply(
        lambda row: extract_relevant_values(
            row['analysis'], 
            row['protein_value'], 
            relevant_anlysis
        ), 
        axis=1
    )
    
    #delimit_value
    rows['delimit_value'] = rows['delimit_value'].apply(extract_fractions)
    rows = rows[ rows['delimit_value'].apply(len) == 6]

    #delimit_value2
    rows = rows[ rows['delimit_value2'].apply(len) == 15]

    #protein_value
    rows = rows.dropna()
    return rows

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


    CNNModel = CNNNetwork().to(device)
    AutoEncoderModel = Autoencoder().to(device)
    CNNModel.load_state_dict(torch.load("networks/convolution_model.pth", weights_only=True))
    AutoEncoderModel.load_state_dict(torch.load("networks/auto_encoder_model.pth", weights_only=True))
    return (AutoEncoderModel,CNNModel)

def get_autoencoder_reconstructions(dataloader, AutoEncoderModel, device):
    AutoEncoderModel.eval()
    reconstructions = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
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
    

def predict(ids: list[int], df: pd.DataFrame) -> pd.DataFrame:
    rows = pre_process_data(ids,df)
    
    X_curve     = np.array(rows['value'].tolist(),         dtype=np.float32)  # (n, 300)
    X_fractions = np.array(rows['delimit_value'].tolist(),  dtype=np.float32)  # (n, 6)
    X_delimit2  = np.array(rows['delimit_value2'].tolist(), dtype=np.float32)  # (n, 15)
    X_protein   = np.array(rows['protein_value'].tolist(),  dtype=np.float32)  # (n, 8)


    X = np.concatenate([X_curve, X_fractions, X_delimit2,X_protein], axis=1)  # (n, 329)
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

    for X, y,_ in dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
        
    (auto_encoder_model, cnn_model) = load_models(device)
    reconstructions = get_autoencoder_reconstructions(dataloader,auto_encoder_model,device)
    rows['cnn_probabilities'] = predict_cnn(dataloader,cnn_model,device)

    X_curves = np.stack(rows['value'])
    squared_errors = np.power((X_curves-reconstructions),2)
    rows['total_squared_error_gamma_region'] = squared_errors.apply(lambda x: np.sum(x[190:265])) / 1000

    logistic_model = joblib.load('logistic_regression.pkl')
    log_probabilities = logistic_model.predict_proba(np.array(rows['total_squared_error_gamma_region']).reshape(-1,1))
    rows['encoder_probabilities'] = log_probabilities[:,1]

    threshold_cnn = 0.25
    threshold_encoder = 0.25

    rows['predictions'] = (rows['cnn_probabilities'] >= threshold_cnn) | (rows['encoder_probabilities'] >= threshold_encoder)

    rows['prob_product'] = np.multiply(rows['cnn_probs'],rows['encoder_probabilities'])

    return rows






