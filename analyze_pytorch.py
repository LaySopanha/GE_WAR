# analyze_pytorch.py (For Phase 1 - 3-Candidate Test)

import os
import random
import numpy as np
import torch
from torchvision import transforms

from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import CNN
from src.utils import evaluate, AES_Sbox

if __name__=="__main__":
    # ======================================================================
    # === CHOOSE WHICH CANDIDATE TO TEST ===
    # ======================================================================
    # Uncomment the line for the model you want to evaluate.
    # Run this script once for each candidate.
    
    CANDIDATE_NAME = "Candidate_A_Shallow_Wide" 
    # CANDIDATE_NAME = "Candidate_B_Deep_Narrow"
    # CANDIDATE_NAME = "Candidate_C_Balanced"
    # ======================================================================

    print(f"--- Evaluating model: {CANDIDATE_NAME} ---")

    # Load the config for the specific candidate
    config_path = f"./Result/CHES_2025_cnn_ID/models/{CANDIDATE_NAME}_config.npy"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}. Did the training for this candidate complete?")
    config = np.load(config_path, allow_pickle=True).item()

    # --- Setup from config ---
    DATASET = config['dataset']
    LEAKAGE = config['leakage']
    POI_START, POI_END = config['poi_start'], config['poi_end']
    POI_WIDTH = POI_END - POI_START
    nb_attacks = 100
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if LEAKAGE == 'ID':
        def leakage_fn(att_plt, k): return AES_Sbox[k ^ int(att_plt)]
        classes = 256
    else: # HW
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]; return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9
    
    # --- Untouchable Block 1 ---
    dataloadertest = Custom_Dataset(root='../', dataset=DATASET, leakage="ID", transform=transforms.Compose([ToTensor_trace()]))
    
    # --- Untouchable Block 2 ---
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    plt_attack = dataloadertest.plt_attack
    
    # --- Rule-Compliant Preprocessing ---
    print(f"Original attack trace shape: {X_attack.shape}")
    X_attack = X_attack[:, POI_START:POI_END]
    print(f"Attack traces manually cropped to shape: {X_attack.shape}")
    
    # We must scale the attack data with the same scaler used for training.
    temp_scaler_loader = Custom_Dataset(root='../', dataset=DATASET, leakage=LEAKAGE, poi_start=POI_START, poi_end=POI_END, train_end=500000)
    X_attack = temp_scaler_loader.scaler_std.transform(X_attack)
    print("Attack traces scaled correctly.")
    
    total_nb_traces_attacks = len(X_attack)

    # --- Load Trained Model ---
    model_path = f"./Result/CHES_2025_cnn_ID/models/{CANDIDATE_NAME}.pth"
    print(f"Loading model from: {model_path}")
    
    search_space = { "layers": config['layers'], "neurons": config['neurons'], "activation": config['activation'], "pooling_types": config['pooling_types'], "pooling_sizes": config['pooling_sizes'], "conv_layers": config['conv_layers'], "filters": config['filters'], "kernels": config['kernels'], "padding": config['padding'] }
    model = CNN(search_space, POI_WIDTH, classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- Evaluate ---
    GE, NTGE = evaluate(device, model, X_attack, plt_attack, correct_key, 
                        leakage_fn=leakage_fn, nb_attacks=nb_attacks,
                        total_nb_traces_attacks=total_nb_traces_attacks,
                        nb_traces_attacks=total_nb_traces_attacks)