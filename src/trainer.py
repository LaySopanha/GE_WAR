import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm

def training_loop(config, model, train_loader, val_loader, device):
    """
    A modern training loop integrated with wandb.
    Saves the best model based on validation loss.
    """
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    else: # Fallback, though sweep only uses Adam
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)
        
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(config.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        tk0 = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for inputs, labels in tk0:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            tk0.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            tk1 = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
            for inputs, labels in tk1:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"*** New best model saved with validation loss: {best_val_loss:.4f} ***")

    print("Finished Training.")
    return model