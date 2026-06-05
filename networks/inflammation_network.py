import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from functions.training import train_loop


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# ── Hjälpfunktioner för CORAL-kodning ───────────────────────────────────────

def to_severity(label: int) -> int:
    match label:
        case 20: return 0
        case 19: return 1
        case 21: return 2
        case 16 | 17 | 18: return 3
        case 22: return 3
        case 23: return 4
        case 24: return 5
        case 25: return 6
        case 26: return 7
        case 27: return 8
        case _:  return 0


def to_coral(label: int, num_classes: int = 9) -> list[int]:
    severity = to_severity(label)
    return [1 if severity > i else 0 for i in range(num_classes - 1)]


def coral_predict_soft(logits: torch.Tensor) -> torch.Tensor:
    """Summerar sigmoid-sannolikheter → predikterad svårighetsgrad."""
    return torch.sigmoid(logits).sum(dim=1)


# ── Nätverksarkitektur ───────────────────────────────────────────────────────

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


# ── Modellklass ──────────────────────────────────────────────────────────────

class InflammationModel:
    def __init__(self, model_path='../models/inflammation_model.pth'):
        self.model_path = model_path
        self.model = InflammationNetwork().to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predikterar inflammationsgrad och skriver till df['inflammation']."""
        X = torch.tensor(np.array(df['value'].tolist(), dtype=np.float32))
        dataloader = DataLoader(TensorDataset(X), batch_size=512)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0][:, 0:300].to(device)
                logits = self.model(inputs)
                probs  = coral_predict_soft(logits)
                all_probs.append(probs.cpu().numpy())

        df['inflammation'] = np.concatenate(all_probs).astype(int)
        return df

    def retrain(self, train_dl, val_dl, epochs=50, patience=5):
        """Tränar om modellen och sparar bästa vikterna."""
        loss_fn   = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_loop(
            train_dl, val_dl, self.model, loss_fn, optimizer, scheduler,
            save_path=self.model_path,
            epochs=epochs,
            patience=patience,
            device=device,
            autoencoder=False
        )
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

    def evaluate(self, dataloader) -> dict:
        """Utvärderar modellen och returnerar MAE och bias."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y, _ in dataloader:
                X = X.to(device)
                logits = self.model(X)
                preds  = coral_predict_soft(logits)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.sum(dim=1).numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        mae  = np.mean(np.abs(all_preds - all_labels))
        bias = np.mean(all_preds - all_labels)
        fn   = ((all_preds < 2) & (all_labels > 4)).sum()

        print(f"MAE:  {mae:.3f}")
        print(f"Bias: {bias:.3f}")
        print(f"FN (pred<2, true>4): {fn} ({100*fn/len(all_preds):.1f}%)")
        for thresh in [1, 2, 3]:
            pct = 100 * (np.abs(all_preds - all_labels) > thresh).sum() / len(all_preds)
            print(f"Error > {thresh} nivåer: {pct:.1f}%")

        return {'mae': mae, 'bias': bias, 'false_negatives': fn}
    



def comment_inflammation(df) -> str:
    severity = df['inflammation'][0]
    crp = df['crp'][0]
    if severity == 0 and crp > 3.0:
        return "Lätt förhöjd halt av CRP som enda tecken på inflammation. "
    match severity:
        case 0: return "Inga tecken på inflammation. "
        case 1: return "Tecken på diskret inflammation. "
        case 2: return "Tecken på lätt inflammation. "
        case 3: return "Tecken lätt-måttlig inflammation. "
        case 4: return "Tecken på måttlig inflammation. "
        case 5: return "Tecken på måttlig-kraftig inflammation. "
        case 6: return "Tecken på kraftig inflammation. "
        case 7: return "Tecken på mycket kraftig inflammation. "