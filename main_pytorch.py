# main_pytorch.py

import os
import random
import numpy as np
import torch
from torchvision import transforms
from copy import deepcopy
import wandb

from src.dataloader import ToTensor_trace, Custom_Dataset, DataAugmentation
from src.net import MLP, CNN, weight_init
from src.trainer import training_loop
from src.utils import evaluate, AES_Sbox

def run_experiment(exp_config):
    run = wandb.init(project="ge-wars-ches2025", config=exp_config, name=exp_config.get('name'))
    config = wandb.config

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset_obj = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage,
                                 poi_start=config.poi_start, poi_end=config.poi_end,
                                 train_end=500000, test_end=100000)
    dataset_obj.split_attack_set_validation_test()
    
    train_transform = transforms.Compose([
        DataAugmentation(shift_prob=0.7, noise_prob=0.7, max_shift=config.max_shift, noise_level=config.noise_level),
        ToTensor_trace()
    ])
    eval_transform = transforms.Compose([ToTensor_trace()])
    
    train_dataset = deepcopy(dataset_obj); train_dataset.transform = train_transform; train_dataset.choose_phase("train")
    val_dataset = deepcopy(dataset_obj); val_dataset.transform = eval_transform; val_dataset.choose_phase("validation")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    if config.leakage == 'ID': classes = 256
    else: classes = 9
    poi_width = config.poi_end - config.poi_start

    search_space = { "layers": config.layers, "neurons": config.neurons, "activation": config.activation, "pooling_types": config.pooling_types, "pooling_sizes": config.pooling_sizes, "conv_layers": config.conv_layers, "filters": config.filters, "kernels": config.kernels, "padding": config.padding }
    model = CNN(search_space, poi_width, classes).to(device)
    weight_init(model, config.kernel_initializer)

    print(f"--- Starting W&B Run: {wandb.run.name} ---")
    model = training_loop(config, model, train_loader, val_loader, device)
    
    print("\n--- Starting Post-Training Evaluation on Validation Set ---")
    model.load_state_dict(torch.load("best_model.pth"))
    
    val_attack_dataset = deepcopy(dataset_obj)
    val_attack_dataset.transform = eval_transform
    val_attack_dataset.choose_phase("validation")
    
    X_attack_val = val_attack_dataset.X
    plt_attack_val = val_attack_dataset.Plaintext
    correct_key = val_attack_dataset.correct_key
    
    if config.leakage == 'ID':
        def leakage_fn(att_plt, k): return AES_Sbox[k ^ int(att_plt)]
    else:
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]; return hw[AES_Sbox[k ^ int(att_plt)]]

    GE, NTGE = evaluate(device, model, X_attack_val, plt_attack_val, correct_key, 
                        leakage_fn=leakage_fn, nb_attacks=50,
                        total_nb_traces_attacks=len(X_attack_val),
                        nb_traces_attacks=len(X_attack_val))

    if NTGE != float('inf'): wandb.run.summary["final_GE_at_max_traces"] = 0
    else: wandb.run.summary["final_GE_at_max_traces"] = GE[-1]
    wandb.run.summary["final_NTGE"] = NTGE
    
    ge_data = [[x+1, y] for (x, y) in enumerate(GE)]
    table = wandb.Table(data=ge_data, columns = ["Traces", "GE"])
    wandb.log({"GE_Curve_Validation" : wandb.plot.line(table, "Traces", "GE", title="Guessing Entropy vs. Traces (Validation)")})
    
    print(f"--- Finished W&B Run: {wandb.run.name} ---")
    wandb.finish()

if __name__ == "__main__":
    # We will run one of our successful candidates from Phase 1 to test the MLOps pipeline
    experiment_config = {
        "name": "Phase2_Test_Candidate_A",
        "epochs": 50, "dataset": "CHES_2025", "leakage": "ID",
        "poi_start": 4119, "poi_end": 4519, "max_shift": 15, "noise_level": 0.05,
        "model_type": "cnn", "layers": 2, "neurons": 256, "activation": "selu",
        "pooling_types": "average_pool", "pooling_sizes": 2, "padding": 0,
        "kernel_initializer": "glorot_uniform", "batch_size": 256, "lr": 1e-3, "optimizer": "Adam",
        # Hyperparameters for Candidate A
        "conv_layers": 2, "filters": 16, "kernels": 12,
    }
    
    run_experiment(experiment_config)