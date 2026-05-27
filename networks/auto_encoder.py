import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from functions.training import train_loop


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class AutoencoderNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),   # 300 -> 150
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),  # 150 -> 75
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), # 75 -> 38
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 38, 16)  # Bottleneck
        )
        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(16, 32 * 38),
            nn.ReLU(),
            nn.Unflatten(1, (32, 38)),
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=0),  # 38 -> 75
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 5, stride=2, padding=2, output_padding=1),   # 75 -> 150
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 7, stride=2, padding=3, output_padding=1),    # 150 -> 300
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.squeeze(1)


def weighted_mse_loss(input, target):
    """Viktar gamma-regionen (index 200+) högre."""
    weights = torch.ones_like(input)
    weights[:, 200:] *= 5.0
    loss = weights * (input - target) ** 2
    return loss.mean()


class AutoencoderModel:
    def __init__(self, model_path='../models/auto_encoder_model.pth'):
        self.model_path = model_path
        self.model = AutoencoderNetwork().to(device)

    def predict(self, df: pd.DataFrame,model_path=None) -> pd.DataFrame:
        """
        Rekonstruerar kurvorna och skriver följande kolumner till df:
          - reconstructions
          - total_squared_error_gamma_region
          - proportion_gamma_region
          - encoder_probability  (via logistisk regression)
        """
        import joblib
        X = np.array(df['value'].tolist(), dtype=np.float32)
        dataloader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=512)
        self.model.load_state_dict(torch.load(model_path or self.model_path, weights_only=True))
        self.model.eval()
        all_reconstructions = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                outputs = self.model(inputs)
                all_reconstructions.append(outputs.cpu().numpy())

        reconstructions = np.concatenate(all_reconstructions, axis=0)
        if reconstructions.ndim == 1:
            reconstructions = reconstructions.reshape(len(df), -1)

        df['reconstructions'] = list(reconstructions)

        X_curves = np.stack(df['value'])
        squared_errors = np.power((X_curves - reconstructions), 2)

        df['total_squared_error_gamma_region'] = np.sum(squared_errors[:, 190:285], axis=1) / 1000
        df['proportion_gamma_region'] = (
            100 * np.sum(squared_errors[:, 190:285], axis=1) /
            np.sum(squared_errors, axis=1)
        )

        logistic_model = joblib.load('../models/logistic_regression.pkl')
        log_probs = logistic_model.predict_proba(
            np.array(df['total_squared_error_gamma_region']).reshape(-1, 1)
        )
        df['encoder_probability'] = log_probs[:, 1]

        return df

    def retrain(self, train_dl, val_dl, epochs=1000, patience=10,model_path=None):
        """Tränar om autoencoder och sparar bästa vikterna."""
        loss_fn   = weighted_mse_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6
                )
        save_path = model_path or self.model_path

        train_loop(
            train_dl, val_dl, self.model, loss_fn, optimizer, scheduler,
            save_path=save_path,
            epochs=epochs,
            patience=patience,
            device=device,
            autoencoder=True
        )
        self.model.load_state_dict(torch.load(save_path, weights_only=True))

def build_dataloaders(train_rows, val_rows, test_rows, batch_sz=512):
    # Autoencoder tränas bara på negativa fall
    train_rows = train_rows[train_rows['label'] == 0]
    val_rows = val_rows[val_rows['label'] == 0]

    X_train = torch.tensor(np.array(train_rows['value'].tolist(), dtype=np.float32))
    X_val   = torch.tensor(np.array(val_rows['value'].tolist(),   dtype=np.float32))
    X_test  = torch.tensor(np.array(test_rows['value'].tolist(),  dtype=np.float32))

    y_train = torch.tensor(np.array(train_rows['label'].values, dtype=np.int64))
    y_val   = torch.tensor(np.array(val_rows['label'].values,   dtype=np.int64))
    y_test  = torch.tensor(np.array(test_rows['label'].values,  dtype=np.int64))

    train_dl = DataLoader(TensorDataset(X_train, y_train, torch.tensor(train_rows['id'].to_numpy())), batch_size=batch_sz, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_val,   y_val,   torch.tensor(val_rows['id'].to_numpy())),   batch_size=batch_sz)
    test_dl  = DataLoader(TensorDataset(X_test,  y_test,  torch.tensor(test_rows['id'].to_numpy())),  batch_size=batch_sz)

    return train_dl, val_dl, test_dl