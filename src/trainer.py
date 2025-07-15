# src/trainer.py

import torch
from torch import nn, optim
from src.net import MLP, CNN, weight_init
from tqdm import tqdm
import wandb
import time

def training_loop(config, model, train_loader, val_loader, device):
    if config['optimizer'] == "Adam":
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    else: # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    wandb.watch(model, criterion, log="all", log_freq=100)
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_corrects = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", leave=False)
        for i, (traces, labels) in enumerate(progress_bar):
            inputs, labels = traces.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()

            if (i + 1) % 100 == 0:
                total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                wandb.log({"gradient_norm": total_norm})

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for traces, labels in val_loader:
                inputs, labels = traces.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        # --- Epoch Logging ---
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_corrects.double() / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1, "train_loss": avg_train_loss, "train_accuracy": avg_train_acc,
            "val_loss": avg_val_loss, "val_accuracy": avg_val_acc, "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            
    wandb.save("best_model.pth")
    return model