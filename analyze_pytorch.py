import os
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

# Import your custom modules as defined in your project structure
from src.dataloader import Custom_Dataset, ToTensor_trace
from src.net import CNN
from src.utils import calculate_snr 
from src.utils import evaluate


if __name__=="__main__":
    # --- Configuration: Must match the training script ---
    # This config is copied from main_pytorch.py to ensure the model
    # and data processing are identical.
    config = {
    "dataset": "CHES_2025",
    "leakage": "ID",
    "num_poi": 100,
    "max_shift": 10,
    "noise_level": 0.03505419099952169,
    "batch_size": 128,
    "lr": 0.001,
    "epochs": 100, 
    "optimizer": "Adam",
    "conv_layers": 3,
    "filters": 8,
    "kernels": 12,
    "activation": "relu",
    "pooling_types": "average_pool",
    "pooling_sizes": 4,
    "padding": 0,
    "layers": 2,
    "neurons": 256,
    "kernel_initializer": "glorot_uniform"
}
    dataset = config['dataset']
    leakage = config['leakage']

    # --- Setup ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Challenge evaluation parameters
    nb_attacks = 100
    nb_traces_attacks = 100000

    # --- Recreate Preprocessing Parameters (POIs and Scaler) ---
    # We must replicate the *logic* from the training script to get the correct
    # POIs and scaler. This requires temporarily loading the profiling data.
    print("Loading profiling data to determine POIs and fit scaler...")
    profiling_dataset = Custom_Dataset(root='../', dataset=config['dataset'], leakage=config['leakage'],
                                     poi_start=0, poi_end=7000,
                                     train_end=500000, test_end=0) # Only need the profiling set

    # Calculate SNR and select top POIs
    snr = calculate_snr(profiling_dataset.X_profiling, profiling_dataset.Y_profiling)
    top_k_indices = np.argsort(snr)[-config['num_poi']:]
    print(f"Identified {len(top_k_indices)} POIs based on SNR.")

    # Fit a StandardScaler on the POI-selected profiling data
    scaler = StandardScaler()
    X_profiling_poi = profiling_dataset.X_profiling[:, top_k_indices]
    scaler.fit(X_profiling_poi)
    print("StandardScaler fitted on profiling data.")
    # We no longer need the profiling_dataset, so it can be cleared from memory
    del profiling_dataset, X_profiling_poi, snr


    ##################please do not touch this code below###################
    dataloadertest = Custom_Dataset(root='./../', dataset=dataset, leakage="ID", #change root to where you download your dataset.
                                                 transform=transforms.Compose([ToTensor_trace()]))
    #########################################################################
    
    # This block is now left as is, since our evaluate function handles it.
    if leakage == 'ID':
        classes = 256
    elif leakage == 'HW':
        classes = 9
    else:
        pass # The original script had this, so we keep it.

    ##################please do not touch this code here###################
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    Y_attack = dataloadertest.Y_attack
    plt_attack = dataloadertest.plt_attack
    num_sample_pts = X_attack.shape[-1]
    #########################################################################

    # --- Apply Preprocessing to the Data from the Fixed Block ---
    # The variable `X_attack` from the block above contains the raw traces.
    # We must now apply the POI selection and scaling we determined earlier.
    print("Applying POI selection and scaling to the attack traces...")
    X_attack_poi = X_attack[:, top_k_indices]
    X_attack_final = scaler.transform(X_attack_poi)


    # --- Model Loading ---
    print("Loading the trained model...")
    model_path = "Result/CHES_2025_cnn_ID/models/fully_trained_model_2.pth"
    
    # Instantiate the model with the same architecture from net.py
    # The poi_width must match the number of selected POIs
    model = CNN(config, config['num_poi'], classes).to(device)
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode


    # --- Evaluation ---
    # The call to evaluate() uses the fully preprocessed data.
    print("\nStarting final evaluation...")
    GE, NTGE = evaluate(
        device=device,
        model=model,
        X_attack=X_attack_final,  # Use the processed attack traces
        plt_attack=plt_attack,    # This is from the fixed block
        correct_key=correct_key,  # This is from the fixed block
        nb_attacks=nb_attacks,
        nb_traces_attacks=nb_traces_attacks,
        batch_size=config['batch_size']
    )
    
    # Final results
    print("\n--- Final Results ---")
    print(f"GE Curve (last 5 points): {GE[-5:]}")
    print(f"Final GE after {nb_traces_attacks} traces: {GE[-1]:.4f}")
    print(f"NTGE: {NTGE}")