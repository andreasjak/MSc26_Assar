from matplotlib import pyplot as plt
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
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
            nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )


        # Tabelldata – fractions(6) + boundaries(12) + proteiner(8) = 26
        self.tabular = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Gemensam klassificerare – CNN-features (256) + tabular (32)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        curve   = x[:, :300].unsqueeze(1)
        tabular = x[:, 300:]

        curve   = self.features(curve)
        avg_pool = torch.mean(curve, dim=2)
        max_pool = torch.max(curve, dim=2).values
        curve_features   = torch.cat([avg_pool, max_pool], dim=1)  # (n, 256)
        tabular_features = self.tabular(tabular)                    # (n, 32)

        combined = torch.cat([curve_features, tabular_features], dim=1)  # (n, 288)
        return self.classifier(combined)


class Comment108Model:
    def __init__(self,
                 model_path='../models/comment_108.pth',
                 scaler_path='../models/scaler.pkl'):
        self.model_path = model_path
        self.model = CNNNetwork().to(device)
        self.scaler = joblib.load(scaler_path)
        self.model.load_state_dict(torch.load(model_path or self.model_path, weights_only=True))
        self.model.to(device)

    def predict(self, df: pd.DataFrame, model_path=None) -> pd.DataFrame:
        """Beräknar P(M-komponent) och skriver till df['comment_108']."""
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

        df['comment_108'] = np.concatenate(all_probs)
        return df
    
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

    def retrain(self, train_dl, val_dl, epochs=10000, patience=5,model_path=None):
        """Tränar om modellen och sparar bästa vikterna."""
        pos_weight = torch.tensor([1.0, 6.0], dtype=torch.float32).to(device)
        loss_fn   = nn.CrossEntropyLoss(weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
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

    def evaluate(self, df,threshold=0.7):
        df = self.predict(df)
        all_probs  = np.array(df['comment_108'])
        all_labels = np.array(df['label'])
        all_preds = (all_probs >= threshold).astype(int)
        print(f"Totala mängd datapunkter: {len(all_probs)}")

        print(classification_report(all_labels, all_preds, target_names=['Negativ', 'Positiv']))
        print(confusion_matrix(all_labels, all_preds))

        # ROC-kurva
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

        # AUC
        roc_auc = auc(fpr, tpr)

        print("AUC:", roc_auc)

        fp = sum((all_labels == 0) & (all_preds == 1))
        fn = sum((all_labels == 1) & (all_preds == 0))
        tn = sum((all_labels == 0) & (all_preds == 0))
        tp = sum((all_labels == 1) & (all_preds == 1))


        fn_rate = fn / sum(all_labels == 1)
        fp_rate = fp / sum(all_labels == 0)
        accuracy = sum(all_preds == all_labels) / len(all_preds)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        print(f"Accuracy:  {100*accuracy:.2f}%")
        print(f"FN-rate:   {100*fn_rate:.2f}%  (farliga missade fall)")
        print(f"FP-rate:   {100*fp_rate:.2f}%  (onödiga larm)")
        print(f"Sensitivitet:   {100*sensitivity:.2f}%  (Sannolikhet att upptäcka ett positivt fall)")
        print(f"Specificitet:   {100*specificity:.2f}%  (Sannolikhet att korrekt klassificera ett negativt fall negativt)")
        print(f"AUC: {roc_auc:.4f}")

        # Rita kurvan
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.show()
        return df