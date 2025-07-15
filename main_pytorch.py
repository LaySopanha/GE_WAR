# main_pytorch.py (Phase 1 - Robust 3-Candidate Test)

import os
import random
import numpy as np
import torch
from torchvision import transforms
from copy import deepcopy

from src.dataloader import ToTensor_trace, Custom_Dataset, DataAugmentation
from src.net import CNN, weight_init
from src.trainer import training_loop # We'll use the non-wandb version for this phase

# This is a simplified trainer for Phase 1, without wandb, for clarity.
# You can also use your MLOps trainer, but this is cleaner for this specific task.
def simple_trainer(config, model, train_loader, val_loader, device):
    from torch import optim
    from tqdm import tqdm
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(config['epochs']):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        for traces, labels in progress_bar:
            inputs, labels = traces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Simple validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for traces, labels in val_loader:
                inputs, labels = traces.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}")
        
    return model

def run_single_training(config):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*50)
    print(f"STARTING RUN: {config['name']}")
    print("="*50)

    dataset_obj = Custom_Dataset(root='../', dataset=config['dataset'], leakage=config['leakage'],
                                 poi_start=config['poi_start'], poi_end=config['poi_end'],
                                 train_end=500000, test_end=100000)
    dataset_obj.split_attack_set_validation_test()
    
    train_transform = transforms.Compose([
        DataAugmentation(max_shift=config['max_shift'], noise_level=config['noise_level']),
        ToTensor_trace()
    ])
    eval_transform = transforms.Compose([ToTensor_trace()])
    
    train_dataset = deepcopy(dataset_obj); train_dataset.transform = train_transform; train_dataset.choose_phase("train")
    val_dataset = deepcopy(dataset_obj); val_dataset.transform = eval_transform; val_dataset.choose_phase("validation")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    if config['leakage'] == 'ID': classes = 256
    else: classes = 9
    poi_width = config['poi_end'] - config['poi_start']
    
    search_space = { "layers": config['layers'], "neurons": config['neurons'], "activation": config['activation'], "pooling_types": config['pooling_types'], "pooling_sizes": config['pooling_sizes'], "conv_layers": config['conv_layers'], "filters": config['filters'], "kernels": config['kernels'], "padding": config['padding'] }
    model = CNN(search_space, poi_width, classes).to(device)
    weight_init(model, config['kernel_initializer'])

    model = simple_trainer(config, model, train_loader, val_loader, device)
    
    save_root = f"./Result/{config['dataset']}_{config['model_type']}_{config['leakage']}/"
    model_root = os.path.join(save_root, "models/")
    os.makedirs(model_root, exist_ok=True)
    
    model_path = os.path.join(model_root, f"{config['name']}.pth")
    np.save(os.path.join(model_root, f"{config['name']}_config.npy"), config)
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete for {config['name']}. Model saved to {model_path}")

if __name__ == "__main__":
    base_config = {
        "epochs": 50, "dataset": "CHES_2025", "leakage": "ID",
        "poi_start": 4119, "poi_end": 4519, "max_shift": 15, "noise_level": 0.05,
        "model_type": "cnn", "layers": 2, "neurons": 256, "activation": "selu",
        "pooling_types": "average_pool", "pooling_sizes": 2, "padding": 0,
        "kernel_initializer": "glorot_uniform", "batch_size": 256, "lr": 1e-3, "optimizer": "Adam",
    }
    
    # Candidate A: "Shallow & Wide"
    config_A = deepcopy(base_config)
    config_A.update({"name": "Candidate_A_Shallow_Wide", "conv_layers": 2, "filters": 16, "kernels": 12})
    
    # Candidate B: "Deep & Narrow"
    config_B = deepcopy(base_config)
    config_B.update({"name": "Candidate_B_Deep_Narrow", "conv_layers": 4, "filters": 4, "kernels": 32})
    
    # Candidate C: "Balanced"
    config_C = deepcopy(base_config)
    config_C.update({"name": "Candidate_C_Balanced", "conv_layers": 3, "filters": 8, "kernels": 24})
    
    candidate_configs = [config_A, config_B, config_C]
    
    for config in candidate_configs:
        run_single_training(config)