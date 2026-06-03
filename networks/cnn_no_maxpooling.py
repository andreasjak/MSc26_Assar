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
        # CNN-del – bara för kurvan (300 punkter)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Tabelldata – fractions(6) + boundaries(12) + proteiner(8) = 26
        self.tabular = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Gemensam klassificerare – CNN-features (128) + tabular (32)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        curve   = x[:, :300].unsqueeze(1)
        tabular = x[:, 300:]

        curve_features   = torch.flatten(self.gap(self.features(curve)), 1)  # (n, 128)
        tabular_features = self.tabular(tabular)                              # (n, 32)

        combined = torch.cat([curve_features, tabular_features], dim=1)      # (n, 160)
        return self.classifier(combined)


class CNNModel:
    def __init__(self,
                 model_path='../models/convolution_model_no_max_pooling.pth',
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