import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from functions.training import train_loop


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class OligoclonalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16,16,kernel_size=7,padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16,16,kernel_size=7,padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16,16,kernel_size=7,padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )


        self.classifier = nn.Sequential(
            nn.Linear(300*16, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        curve   = x[:, :300].unsqueeze(1)

        curve_features   = torch.flatten(self.features(curve), 1)  # (n, 128)

        return self.classifier(curve_features)


class OligoclonalModel:
    def __init__(self, 
                 model_path='../models/oligoclonal_model.pth',
                 scaler_path='../models/global_scaler_ln_300.pkl'):
        self.model_path = model_path
        self.model = OligoclonalNetwork().to(device)
        self.scaler = joblib.load(scaler_path)
        self.model.to(device)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Beräknar P(oligoklonalt) och skriver till df['oligoclonal_probability']."""
        X = df['value'].to_list()
        X = self.scaler.transform(X)
        X = torch.tensor(np.array(X, dtype=np.float32))
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        dataloader = DataLoader(TensorDataset(X), batch_size=512)
        self.model.eval()
        
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                logits = self.model(inputs)
                probs  = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy())

        df['oligoclonal_probability'] = np.concatenate(all_probs)
        return df
    
    def evaluate(self, df,threshold=0.7):
        df = self.predict(df)
        all_probs  = np.array(df['oligoclonal_probability'])
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

    def retrain(self, train_dl, val_dl, epochs=10000, patience=5,model_path = None):
        """Tränar om modellen och sparar bästa vikterna."""
        self.model = OligoclonalNetwork().to(device)
        pos_weight = torch.tensor([1.0, 6], dtype=torch.float32).to(device)
        loss_fn   = nn.CrossEntropyLoss(weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=4,
                min_lr=1e-6
            )

        path=model_path or self.model_path
        best_auc = train_loop(
            train_dl, val_dl, self.model, loss_fn, optimizer, scheduler,
            save_path=path,
            epochs=epochs,
            patience=patience,
            device=device,
            autoencoder=False,
        )
        self.model.load_state_dict(torch.load(path, weights_only=True))
        return best_auc
    
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
            self.model = OligoclonalNetwork().to(device)
            # Splitta datan för denna fold
            fold_train_df = train_df.iloc[train_idx]
            fold_val_df = train_df.iloc[val_idx]
            
            # Bygg dataloaders
            fold_train_dl = self.build_dataloader(fold_train_df)
            fold_val_dl = self.build_dataloader(fold_val_df)
            
            
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
                'val_auc': metrics['auc']
            }
            kfold_history.append(fold_results)
            
            print(f"-> Slutresultat Fold {fold+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {metrics['auc']:.4f}")
            
        # --- Sammanställning i Pandas ---
        df_metrics = pd.DataFrame(kfold_history)
        
        # Spara till CSV-fil
        csv_save_path = '../models/oligoclonal_kfold_metrics.csv'
        df_metrics.to_csv(csv_save_path, index=False)
        
        print("\n" + "="*50)
        print(" K-FOLD SLUTRESULTAT (SPARAT TILL PANDAS / CSV)")
        print("="*50)
        print(df_metrics.to_string(index=False))
        print("-"*50)
        print(f"Genomsnittlig Val AUC:  {df_metrics['val_auc'].mean():.4f} (± {df_metrics['val_auc'].std():.4f})")
        print(f"Genomsnittlig Val Loss: {df_metrics['val_loss'].mean():.4f} (± {df_metrics['val_loss'].std():.4f})")
        print("="*50)
        
        return df_metrics

    def build_dataloader(self,rows, batch_sz=512):
        X = self.scaler.transform(rows['value'].tolist())
        X = np.array(X,dtype=np.float32)
        
        y = torch.tensor(rows['label'].values, dtype=torch.long)
        ids = torch.tensor(rows['row_id'].to_numpy(), dtype=torch.long)
        X = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X, y, ids)

        return DataLoader(dataset, batch_size=batch_sz, shuffle=True)


