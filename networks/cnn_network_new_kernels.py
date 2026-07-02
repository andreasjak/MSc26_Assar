from sklearn.model_selection import StratifiedKFold
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
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=12, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=6, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=2, padding=1),
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
                 model_path='../models/convolution_model_kernel.pth',
                 scaler_path='../models/scaler.pkl'):
        self.model_path = model_path
        self.model = CNNNetwork().to(device)
        self.scaler = joblib.load(scaler_path)
        #self.model.load_state_dict(torch.load(model_path or self.model_path, weights_only=True))

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
    
    def reset_weights(self):
        self.model = CNNNetwork().to(device)

    
    def retrain_with_k_fold(self, train_df: pd.DataFrame, k=10, epochs=100, patience=15):
        """
        Tränar modellen med Stratified K-Fold Cross-Validation.
        Sparar train_loss, val_loss, val_acc och val_auc för varje fold i en Pandas DataFrame och CSV.
        """

        # Vi importerar validate_epoch explicit här om den ligger i samma modul som train_loop
        from functions.training import train_loop, validate_epoch 
                
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        # Lista för att samla historik (skapas till en Pandas DF i slutet)
        kfold_history = []
        
        X_dummy = np.zeros(len(train_df))
        y_dummy = train_df['label'].values
        
        print(f"--- Startar {k}-Fold Cross Validation ---")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_dummy, y_dummy)):
            print(f"\n" + "="*40)
            print(f" FOLD {fold + 1}/{k}")
            print("="*40)
            self.model = CNNNetwork().to(device)
            # Splitta datan för denna fold
            fold_train_df = train_df.iloc[train_idx]
            fold_val_df = train_df.iloc[val_idx]
            
            # Bygg dataloaders
            fold_train_dl, fold_val_dl, _ = build_dataloaders(fold_train_df, fold_val_df, fold_val_df)
            
            
            # Definiera unikt filnamn för den bästa modellen i denna fold
            fold_model_path = self.model_path.replace('.pth', f'_fold{fold+1}.pth')
            
            # Återställ optimizer och scheduler precis som i din vanliga retrain
            pos_weight = torch.tensor([1.0, 6.0], dtype=torch.float32).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=pos_weight)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
            )
            
            # Kör din befintliga träningsloop (den sköter tidig stoppning och sparar bäst vikter till fold_model_path)
            train_loop(
                train_dl=fold_train_dl,
                val_dl=fold_val_dl,
                model=self.model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                save_path=fold_model_path,
                epochs=epochs,
                patience=patience,
                device=device,
                autoencoder=False
            )
            
            # --- Utvärdering av den bästa modellen för denna fold ---
            print(f"\nUtvärderar bästa modellen för Fold {fold+1}...")
            
            # Ladda in de absolut bästa vikterna som sparades under körningen
            self.model.load_state_dict(torch.load(fold_model_path, weights_only=True))
            
            # Kör valideringen en sista gång för att extrahera slutgiltig förlust och metrics
            metrics, val_loss = validate_epoch(fold_val_dl, self.model, loss_fn, device, autoencoder=False)
            
            # Vi kör även validate_epoch på träningsdatan för att få ut slutgiltig train_loss efter tidig stoppning
            _, train_loss = validate_epoch(fold_train_dl, self.model, loss_fn, device, autoencoder=False)
            
            # Spara ner resultaten i vår lista
            fold_results = {
                'fold': fold + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': metrics['accuracy'],
                'val_auc': metrics['auc'],
                'val_spec': metrics['specificity'],
                'val_sens': metrics['sensitivity'],
                'tn': metrics['tn'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'tp': metrics['tp']
            }
            kfold_history.append(fold_results)
            
            print(f"-> Slutresultat Fold {fold+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {metrics['auc']:.4f}")
            
        # --- Sammanställning i Pandas ---
        df_metrics = pd.DataFrame(kfold_history)
        
        # Spara till CSV-fil
        csv_save_path = '../models/cnn_different_kernel_kfold_metrics.csv'
        df_metrics.to_csv(csv_save_path, index=False)
        
        print("\n" + "="*50)
        print(" K-FOLD SLUTRESULTAT (SPARAT TILL PANDAS / CSV)")
        print("="*50)
        print(df_metrics.to_string(index=False))
        print("-"*50)
        print(f"Genomsnittlig Val AUC:  {df_metrics['val_auc'].mean():.4f} (± {df_metrics['val_auc'].std():.4f})")
        print(f"Genomsnittlig Val accuracy:  {df_metrics['val_accuracy'].mean():.4f} (± {df_metrics['val_accuracy'].std():.4f})")
        print(f"Genomsnittlig Val Loss: {df_metrics['val_loss'].mean():.4f} (± {df_metrics['val_loss'].std():.4f})")
        print(f"Genomsnittlig Val specificity: {df_metrics['val_spec'].mean():.4f} (± {df_metrics['val_spec'].std():.4f})")
        print(f"Genomsnittlig Val sensitivity: {df_metrics['val_sens'].mean():.4f} (± {df_metrics['val_sens'].std():.4f})")
        print(f"Genomsnittlig Val tn: {df_metrics['tn'].mean():.2f} (± {df_metrics['tn'].std():.2f})")
        print(f"Genomsnittlig Val fp: {df_metrics['fp'].mean():.2f} (± {df_metrics['fp'].std():.2f})")
        print(f"Genomsnittlig Val fn: {df_metrics['fn'].mean():.2f} (± {df_metrics['fn'].std():.2f})")
        print(f"Genomsnittlig Val tp: {df_metrics['tp'].mean():.2f} (± {df_metrics['tp'].std():.2f})")
        print("="*50)
        
        return df_metrics

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
    
def build_dataloader(rows, batch_sz=512):
    protein_cols = [a.col for a in ANALYTES[:8]]
    rows = rows.dropna(subset=protein_cols)

    X = np.concatenate([
        np.array(rows['value'].tolist(),      dtype=np.float32),
        np.array(rows['fractions'].tolist(),  dtype=np.float32),
        np.array(rows['boundaries'].tolist(), dtype=np.float32),
        np.array(rows[protein_cols].values,   dtype=np.float32),
    ], axis=1)

    scaler = joblib.load('../models/scaler.pkl')
    X[:, 300:] = scaler.transform(X[:, 300:])

    y = torch.tensor(rows['label'].values, dtype=torch.long)
    ids = torch.tensor(rows['row_id'].to_numpy(), dtype=torch.long)
    X = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X, y, ids)

    return DataLoader(dataset, batch_size=batch_sz, shuffle=True)

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

    train_dl = DataLoader(TensorDataset(X_train, y_train, torch.tensor(train_rows['row_id'].to_numpy())), batch_size=batch_sz, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_val,   y_val,   torch.tensor(val_rows['row_id'].to_numpy())),   batch_size=batch_sz)
    test_dl  = DataLoader(TensorDataset(X_test,  y_test,  torch.tensor(test_rows['row_id'].to_numpy())),  batch_size=batch_sz)

    return train_dl, val_dl, test_dl
    