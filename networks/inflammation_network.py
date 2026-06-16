import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from functions.analyte import ANALYTES
from functions.training import train_loop_coral

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
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # CRP, haptoglobin, albumin, alpha-1, alpha-2 – mest relevanta
        self.tabular = nn.Sequential(
            nn.Linear(26, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 8)  # CORAL-trösklar
        )

    def forward(self, x):
        curve   = x[:, :300].unsqueeze(1)
        tabular = x[:, 300:]
        curve   = self.features(curve)
        avg_pool = torch.mean(curve, dim=2)
        max_pool = torch.max(curve, dim=2).values
        curve_features   = torch.cat([avg_pool, max_pool], dim=1)  # (n, 256)
        tabular_features = self.tabular(tabular)                    # (n, 32)
        combined = torch.cat([curve_features, tabular_features], dim=1)
        return self.classifier(combined)


# ── Modellklass ──────────────────────────────────────────────────────────────

class InflammationModel:
    def __init__(self, model_path='../models/inflammation_model.pth'):
        self.model_path = model_path
        self.model = InflammationNetwork().to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predikterar inflammationsgrad och skriver till df['inflammation']."""
        protein_cols = [a.col for a in ANALYTES[:8]]
        X = np.concatenate([
            np.array(df['value'].tolist(),      dtype=np.float32),  # (n, 300)
            np.array(df['fractions'].tolist(),  dtype=np.float32),  # (n, 6)
            np.array(df['boundaries'].tolist(), dtype=np.float32),  # (n, 12)
            np.array(df[protein_cols].values,   dtype=np.float32),  # (n, 8)
        ], axis=1)
        
        scaler = joblib.load('../models/scaler.pkl')
        X[:, 300:] = scaler.transform(X[:, 300:])
        
        dataloader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=512)  # Fix 2
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)  # Fix 1 – skicka hela X, inte bara :300
                logits = self.model(inputs)
                probs  = coral_predict_soft(logits)
                all_probs.append(probs.cpu().numpy())
        
        df['inflammation'] = np.concatenate(all_probs).astype(int)
        return df

    def retrain(self, train_dl, val_dl, epochs=50, patience=10):
        """Tränar om modellen och sparar bästa vikterna."""
        loss_fn   = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
        train_loop_coral(
            train_dl, val_dl, self.model, loss_fn, optimizer, scheduler,
            save_path=self.model_path,
            epochs=epochs,
            patience=patience,
            device=device,
        )
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))

    def evaluate_inflammation(self, test_dl):
        self.model.eval()
        all_preds, all_true = [], []
        
        with torch.no_grad():
            for X, y, _ in test_dl:
                X = X.to(device)
                logits = self.model(X)
                pred_severity = coral_predict_soft(logits).cpu().numpy()
                true_severity = y.sum(dim=1).numpy()
                all_preds.extend(pred_severity)
                all_true.extend(true_severity)
        
        all_preds = np.array(all_preds)
        all_true  = np.array(all_true)
        
        mae  = np.abs(all_preds - all_true).mean()
        acc = (all_preds.round().astype(int) == all_true.astype(int)).mean() * 100 
        off1 = (np.abs(all_preds - all_true) <= 1).mean() * 100  # Inom ±1 grad
        off2 = (np.abs(all_preds - all_true) <= 2).mean() * 100  # Inom ±1 grad


        print(f"MAE:          {mae:.3f}")
        print(f"Exakt rätt:   {acc:.1f}%")
        print(f"Inom ±1 grad: {off1:.1f}%")
        print(f"Inom ±2 grad: {off2:.1f}%")
        print(f"The bias is: {np.mean(all_preds - all_true)}")

        for i in range(10):
            no_predicted = sum(np.abs(all_preds - i) < 1)
            true_amount = sum(np.abs(all_true - i < 1))
            print(f"inflammation level: {i} | predicted amount within 1 unit: {no_predicted} | true amount within 1 unit: {true_amount}")

        
        # Konfusionsmatris
        labels = [f"Grad {i}" for i in range(8)]
        cm = confusion_matrix(all_true.astype(int), all_preds.round().astype(int), labels=list(range(8)))
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predikterad')
        ax.set_ylabel('Verklig')
        ax.set_title('Konfusionsmatris – Inflammation')
        plt.colorbar(im)
        
        # Annotation
        for i in range(8):
            for j in range(8):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        plt.tight_layout()
        plt.show()
        
        # Distribution av fel
        errors = all_preds - all_true
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(errors, bins=range(int(errors.min())-1, int(errors.max())+2), 
                color='steelblue', alpha=0.8, align='left')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Prediktionsfel (pred - verklig)')
        ax.set_ylabel('Antal')
        ax.set_title('Distribution av prediktionsfel')
        plt.tight_layout()
        plt.show()


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

    y_train = torch.tensor(np.array(train_rows['label'].tolist(), dtype=np.float32))  # (n, 8)
    y_val   = torch.tensor(np.array(val_rows['label'].tolist(),   dtype=np.float32))  # (n, 8)
    y_test  = torch.tensor(np.array(test_rows['label'].tolist(),  dtype=np.float32))  # (n, 8)

    X_train, X_val, X_test = map(torch.tensor, [X_train, X_val, X_test])

    train_dl = DataLoader(TensorDataset(X_train, y_train, torch.tensor(train_rows['row_id'].to_numpy())), batch_size=batch_sz, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_val,   y_val,   torch.tensor(val_rows['row_id'].to_numpy())),   batch_size=batch_sz)
    test_dl  = DataLoader(TensorDataset(X_test,  y_test,  torch.tensor(test_rows['row_id'].to_numpy())),  batch_size=batch_sz)

    return train_dl, val_dl, test_dl
    



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