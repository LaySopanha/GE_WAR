# src/trainer.py
import torch, time, os
from torch import nn, optim
from tqdm import tqdm
from src.utils import evaluate

def training_loop(config, model, train_loader, val_loader, device, run):
    if config['optimizer'] == "Adam": optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    else: optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    # patience = config.get('early_stopping_patience', 10)  # Default patience of 10

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for i, (traces, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)):
            inputs, labels = traces.to(device), labels.squeeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss, val_corrects = 0.0, 0
        if val_loader:
            with torch.no_grad():
                for traces, labels in tqdm(val_loader, desc=f"Validating...", leave=False):
                    inputs, labels = traces.to(device), labels.squeeze(1).to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            avg_val_acc = val_corrects.double() / len(val_loader.dataset)
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

            if run:
                run.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_acc": avg_val_acc,
                    "lr": scheduler.get_last_lr()[0]
                })
            
            # if avg_val_loss < best_val_loss:
            #     best_val_loss = avg_val_loss
            #     torch.save(model.state_dict(), "best_model.pth")
            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1
        
        scheduler.step()

        # if epochs_no_improve >= patience:
        #     print(f"Early stopping triggered after {epoch + 1} epochs.")
        #     break
    torch.save(model.state_dict(), "best_model.pth")        
    return model


def attack_driven_training_loop(config, model, train_loader, device, run, X_attack, plt_attack, correct_key, leakage_fn, fold=0):
    if config['optimizer'] == "Adam": optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    else: optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)
    criterion = nn.CrossEntropyLoss()
    
    best_final_ge = float('inf')
    best_ntge = float('inf')
    epochs_no_improve = 0
    patience = config.get('early_stopping_patience', 20)
    min_epochs = config.get('min_epochs', 50)  # Minimum epochs before stopping
    best_model_path = f"best_model_fold_{fold}.pth"

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for i, (traces, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)):
            inputs, labels = traces.to(device), labels.squeeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- Attack-Driven Evaluation ---
        GE, NTGE, final_ge = evaluate(device, model, X_attack, plt_attack, correct_key,
                                      leakage_fn=leakage_fn, nb_attacks=100,
                                      total_nb_traces_attacks=config["num_traces_attack"],
                                      nb_traces_attacks=config["num_traces_attack"])
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Final GE: {final_ge:.2f}, NTGE: {NTGE}")

        if run:
            run.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "final_GE": final_ge,
                "NTGE": NTGE,
                "lr": scheduler.get_last_lr()[0]
            })

        # Best practice early stopping
        if final_ge < best_final_ge or (final_ge == best_final_ge and NTGE < best_ntge):
            best_final_ge = final_ge
            best_ntge = NTGE
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Final GE: {best_final_ge:.2f} and NTGE: {best_ntge}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        scheduler.step()
    
    # print(f"Training finished. Loading best model from {best_model_path}")
    # model.load_state_dict(torch.load(best_model_path))
    return model
