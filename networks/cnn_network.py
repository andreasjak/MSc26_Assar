from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from functions.analyte import ANALYTES
from functions.training import train_loop


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNNModel:
    def __init__(self,
                 model_path='../models/convolution_model.pth',
                 scaler_path='../models/scaler.pkl'):
        self.model_path = model_path
        self.model = CNNNetwork().to(device)
        self.scaler = joblib.load(scaler_path)

    def predict(self, df: pd.DataFrame, model_path=None) -> pd.DataFrame:
        """Beräknar P(M-komponent) och skriver till df['cnn_probability']."""
        X = self._build_X(df)
        dataloader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=512)
        self.model.load_state_dict(torch.load(model_path or self.model_path, weights_only=True))
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy())

        df['cnn_probability'] = np.concatenate(all_probs)
        return df

    def retrain(self, train_dl, val_dl, epochs=10000, patience=5,model_path=None):
        """Tränar om modellen och sparar bästa vikterna."""
        pos_weight = torch.tensor([1.0, 6.0], dtype=torch.float32).to(device)
        loss_fn   = nn.CrossEntropyLoss(weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=4,
                min_lr=1e-6
            )
        save_path=model_path or self.model_path

        train_loop(
            train_dl, val_dl, self.model, loss_fn, optimizer, scheduler,
            save_path=save_path,
            epochs=epochs,
            patience=patience,
            device=device,
            autoencoder=False
        )
        self.model.load_state_dict(torch.load(save_path, weights_only=True))

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        protein_cols = [a.col for a in ANALYTES[:8]]
        X = np.concatenate([
            np.array(df['value'].tolist(),      dtype=np.float32),  # (n, 300)
            np.array(df['fractions'].tolist(),  dtype=np.float32),  # (n, 6)
            np.array(df['boundaries'].tolist(), dtype=np.float32),  # (n, 12)
            np.array(df[protein_cols].values,   dtype=np.float32),  # (n, 8)
        ], axis=1)
        X[:, 300:] = self.scaler.transform(X[:, 300:])
        return X
    
def build_dataloaders(train_rows, val_rows, test_rows, batch_sz=512):
    protein_cols = [a.col for a in ANALYTES[:8]]
    train_rows = train_rows.dropna(subset=protein_cols)
    val_rows = val_rows.dropna(subset=protein_cols)
    test_rows = test_rows.dropna(subset=protein_cols)

    def build_X(rows):
        return np.concatenate([
            np.array(rows['value'].tolist(),      dtype=np.float32),  # (n, 300)
            np.array(rows['fractions'].tolist(),  dtype=np.float32),  # (n, 6)
            np.array(rows['boundaries'].tolist(), dtype=np.float32),  # (n, 12)
            np.array(rows[protein_cols].values,   dtype=np.float32),  # (n, 8)
        ], axis=1)

    X_train = build_X(train_rows)
    X_val   = build_X(val_rows)
    X_test  = build_X(test_rows)

    # Skala bara features efter index 300, fit bara på träning
    scaler = StandardScaler()
    X_train[:, 300:] = scaler.fit_transform(X_train[:, 300:])
    X_val[:, 300:]   = scaler.transform(X_val[:, 300:])
    X_test[:, 300:]  = scaler.transform(X_test[:, 300:])
    joblib.dump(scaler, '../models/scaler.pkl')

    y_train = torch.tensor(np.array(train_rows['label'].values, dtype=np.int64))
    y_val   = torch.tensor(np.array(val_rows['label'].values,   dtype=np.int64))
    y_test  = torch.tensor(np.array(test_rows['label'].values,  dtype=np.int64))

    X_train, X_val, X_test = map(torch.tensor, [X_train, X_val, X_test])

    train_dl = DataLoader(TensorDataset(X_train, y_train, torch.tensor(train_rows['id'].to_numpy())), batch_size=batch_sz, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_val,   y_val,   torch.tensor(val_rows['id'].to_numpy())),   batch_size=batch_sz)
    test_dl  = DataLoader(TensorDataset(X_test,  y_test,  torch.tensor(test_rows['id'].to_numpy())),  batch_size=batch_sz)

    return train_dl, val_dl, test_dl