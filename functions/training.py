from sklearn.metrics import roc_auc_score
import torch
import numpy as np


def train_epoch(dataloader, model, loss_fn, optimizer, device, autoencoder=False):
    """En träningsepok. autoencoder=True betyder att target är X själv."""
    model.train()
    total_loss = 0
    for batch in dataloader:
        X = batch[0].to(device)
        y = X if autoencoder else batch[1].to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(dataloader, model, loss_fn, device, autoencoder=False):
    """En valideringsepok."""
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for X,y,_ in dataloader:
            if autoencoder:
                y = X
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if not autoencoder:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                probs = torch.softmax(pred, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y.cpu().numpy().astype(int))
    test_loss = test_loss / num_batches
    if not autoencoder:
        correct = correct / size
        all_preds = (np.array(all_probs) >= 0.5).astype(int)
        all_targets = np.array(all_targets)
        fp = sum((all_preds == 1) & (all_targets == 0))
        fn = sum((all_preds == 0) & (all_targets == 1))
        tn = sum((all_preds == 0) & (all_targets == 0))
        tp = sum((all_preds == 1) & (all_targets == 1))

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        auc = roc_auc_score(all_targets, all_probs)
        return ({"accuracy": 100 * correct,"auc": auc, "specificity": specificity, "sensitivity": sensitivity, "tn": tn, "fp": fp, "fn": fn, "tp": tp},test_loss)

    return (None,test_loss)


def train_loop(
    train_dl,
    val_dl,
    model,
    loss_fn,
    optimizer,
    scheduler,
    save_path,
    epochs=1000,
    patience=5,
    device='mps',
    autoencoder=False,
):
    """
    Gemensam träningsloop för CNN och autoencoder.
    
    Args:
        train_dl:    DataLoader för träning
        val_dl:      DataLoader för validering
        model:       PyTorch-modell
        loss_fn:     förlustfunktion
        optimizer:   optimizer
        scheduler:   LR-scheduler
        save_path:   sökväg för att spara bästa modell
        epochs:      max antal epoker
        patience:    early stopping – antal epoker utan förbättring
        device:      'cpu', 'cuda', 'mps' etc
        autoencoder: True om target = input (rekonstruktion)
    """
    no_improve = 0
    metrics,val_loss = validate_epoch(val_dl,model,loss_fn,device,autoencoder)

    best_auc = metrics['auc']
    best_val_loss = val_loss

    for t in range(epochs):
        train_loss = train_epoch(train_dl, model, loss_fn, optimizer, device, autoencoder)
        metrics,val_loss   = validate_epoch(val_dl, model, loss_fn, device, autoencoder)
        scheduler.step(train_loss)
        auc = metrics['auc']

        if auc > best_auc:
            torch.save(model.state_dict(), save_path)
            print(f"  -> ny bästa modell sparad till {save_path}")
            best_auc = auc

        if not autoencoder:
            print(
                f"Epoch {t:>3} | "
                f"train: {train_loss:.4f} | "
                f"val: {val_loss:.4f} | "
                f"acc: {metrics['accuracy']:.2f}% | "
                f"AUC: {metrics['auc']:.3f}  | "
                f"LR: {optimizer.param_groups[0]['lr']}"
            )
        else:
            print(f"Epoch {t:>3} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping efter {t+1} epoker.")
                break

    print("Klar!")

def train_loop_coral(train_dl, val_dl, model, loss_fn, optimizer, scheduler,
                     save_path, epochs=1000, patience=5, device='cpu'):
    best_val_loss = float('inf')
    no_improve = 0

    for t in range(epochs):
        model.train()
        train_loss = 0
        for X, y, _ in train_dl:
            X, y = X.to(device), y.to(device).float()
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss, mae = 0, 0
        with torch.no_grad():
            for X, y, _ in val_dl:
                X, y = X.to(device), y.to(device).float()
                logits = model(X)
                val_loss += loss_fn(logits, y).item()
                pred_severity = torch.sigmoid(logits).sum(dim=1).round()
                true_severity = y.sum(dim=1)
                mae += torch.abs(pred_severity - true_severity).mean().item()
        val_loss /= len(val_dl)
        mae      /= len(val_dl)

        scheduler.step(val_loss)
        print(f"Epoch {t:>3} | train: {train_loss:.4f} | val: {val_loss:.4f} | MAE: {mae:.3f} | lr: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> ny bästa modell sparad")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping efter {t+1} epoker.")
                break

    print("Klar!")