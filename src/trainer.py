# src/trainer.py
import torch, time, os
from torch import nn, optim
from tqdm import tqdm

def training_loop(config, model, train_loader, val_loader, device):
    if config['optimizer'] == "Adam": optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    else: optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        for i, (traces, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)):
            inputs, labels = traces.to(device), labels.squeeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss, val_corrects = 0.0, 0
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
        print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
    return model
