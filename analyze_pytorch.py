# analyze_pytorch.py (Final, Corrected Version for Phase 1)

import os
import random
import numpy as np
import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from src.dataloader import Custom_Dataset, ToTensor_trace
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
    
   
    # --- Load Data and Preprocess Correctly ---
    
    # Step 1: Load the raw attack traces using the untouchable block's logic
    # We create a temporary dataloader object just to get the data arrays
    # Note: We are using leakage="ID" here as specified in the untouchable block
    # but the Y_attack labels are not used in the final evaluation logic anyway.
   

    ##################please do not touch this code below###################
    dataloadertest = Custom_Dataset(root='./../', dataset=DATASET, leakage="ID", #change root to where you download your dataset.
                                                 transform=transforms.Compose([ToTensor_trace()]))
    #########################################################################

    if LEAKAGE == 'ID':
            def leakage_fn(att_plt, k): return AES_Sbox[k ^ int(att_plt)]
            classes = 256
    else:
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]; return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9

   ##################please do not touch this code here###################
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    Y_attack = dataloadertest.Y_attack
    plt_attack = dataloadertest.plt_attack
    num_sample_pts = X_attack.shape[-1]
    #########################################################################
    
    # Manually apply the POI to the attack traces loaded by the dataloader.
    # This is necessary because the untouchable block does not pass the POI to the Custom_Dataset.
    print(f"Manually applying POI to attack traces: {POI_START} to {POI_END}")
    X_attack = X_attack[:, :, POI_START:POI_END]


    # Step 2: The data from the dataloader is already 3D. We need to reshape it for the scaler.
    num_traces, _, num_points = X_attack.shape
    X_attack_2d = X_attack.reshape(num_traces, num_points)

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

    # Step 4: Apply the correctly fitted scaler to our 2D attack traces
    X_attack_scaled_2d = scaler.transform(X_attack_2d)
    print("Attack traces scaled correctly.")

    # Step 5: Reshape the scaled data back to 3D for the model
    X_attack_final = X_attack_scaled_2d.reshape(num_traces, 1, num_points)
    
    total_nb_traces_attacks = len(X_attack_final)

    # --- Load Trained Model ---
    model_path = f"./Result/CHES_2025_cnn_ID/models/{CANDIDATE_NAME}.pth"
    print(f"Loading model from: {model_path}")
    
    search_space = { "layers": config['layers'], "neurons": config['neurons'], "activation": config['activation'], "pooling_types": config['pooling_types'], "pooling_sizes": config['pooling_sizes'], "conv_layers": config['conv_layers'], "filters": config['filters'], "kernels": config['kernels'], "padding": config['padding'] }
    model = CNN(search_space, POI_WIDTH, classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- Evaluate ---
    GE_curve, NTGE = evaluate(device, model, X_attack_final, plt_attack, correct_key, 
                              nb_attacks=nb_attacks,
                              nb_traces_attacks=total_nb_traces_attacks)

    final_ge = GE_curve[-1] if len(GE_curve) > 0 else float('inf')
    print("\n--- Evaluation Complete ---")
    print(f"Final Guessing Entropy (GE) at {total_nb_traces_attacks} traces: {final_ge}")
    print(f"Number of Traces to Guess Entropy (NTGE): {NTGE}")
    print("---------------------------\n")
