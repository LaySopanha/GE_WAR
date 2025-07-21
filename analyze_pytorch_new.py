# analyze_pytorch.py (Final, Corrected Version for Phase 1)

import os
import random
import numpy as np
import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

# We only need CNN and evaluate from our modules
from src.net import CNN
from src.utils import evaluate, AES_Sbox, load_ctf_2025

if __name__=="__main__":
    # ======================================================================
    # === CHOOSE WHICH CANDIDATE TO TEST ===
    # ======================================================================
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
    else:
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]; return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9
    
    # --- Load Data and Preprocess Correctly ---
    
    # Step 1: Load the raw attack traces using the untouchable block's logic
    # We create a temporary dataloader object just to get the data arrays
    # Note: We are using leakage="ID" here as specified in the untouchable block
    # but the Y_attack labels are not used in the final evaluation logic anyway.
    from src.dataloader import Custom_Dataset, ToTensor_trace
    temp_dataloader_for_attack_data = Custom_Dataset(root='../', dataset=DATASET, leakage="ID", transform=transforms.Compose([ToTensor_trace()]))
    temp_dataloader_for_attack_data.split_attack_set_validation_test()
    temp_dataloader_for_attack_data.choose_phase("test")
    X_attack_full = temp_dataloader_for_attack_data.X_attack_test
    plt_attack = temp_dataloader_for_attack_data.plt_attack_test
    correct_key = temp_dataloader_for_attack_data.correct_key

    # Step 2: Manually crop the attack traces to our POI
    X_attack_cropped = X_attack_full[:, POI_START:POI_END]
    print(f"Attack traces manually cropped to shape: {X_attack_cropped.shape}")

    # Step 3: Get the correct scaler. We MUST use the same scaling as was used in training.
    # To do this, we load the cropped *profiling* data and fit a scaler to it.
    print("Fitting scaler on cropped profiling data...")
    (X_profiling_cropped, _), _, _, _ = load_ctf_2025(
        '../Dataset/CHES_2025/CHES_Challenge.h5',
        leakage_model=LEAKAGE,
        train_end=500000,
        poi_start=POI_START,
        poi_end=POI_END
    )
    scaler = StandardScaler()
    scaler.fit(X_profiling_cropped)

    # Step 4: Apply the correctly fitted scaler to our cropped attack traces
    X_attack_final = scaler.transform(X_attack_cropped)
    print("Attack traces scaled correctly.")
    
    total_nb_traces_attacks = len(X_attack_final)

    # --- Load Trained Model ---
    model_path = f"./Result/CHES_2025_cnn_ID/models/{CANDIDATE_NAME}.pth"
    print(f"Loading model from: {model_path}")
    
    search_space = { "layers": config['layers'], "neurons": config['neurons'], "activation": config['activation'], "pooling_types": config['pooling_types'], "pooling_sizes": config['pooling_sizes'], "conv_layers": config['conv_layers'], "filters": config['filters'], "kernels": config['kernels'], "padding": config['padding'] }
    model = CNN(search_space, POI_WIDTH, classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- Evaluate ---
    GE, NTGE = evaluate(device, model, X_attack_final, plt_attack, correct_key, 
                        leakage_fn=leakage_fn, nb_attacks=nb_attacks,
                        total_nb_traces_attacks=total_nb_traces_attacks,
                        nb_traces_attacks=total_nb_traces_attacks)