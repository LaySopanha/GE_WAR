# evaluate_all_models_QUICK.py
import numpy as np
import os
import torch
import csv
import time
from src.net import MLP, CNN
# Make sure to import all your utility functions
from src.utils import AES_Sbox, calculate_HW, normalize_trace, align_traces, evaluate
from src.dataloader import load_ctf_2025

# This script can be run on a CPU.
device = torch.device("cpu")

# --- Configuration (should match your training run) ---
dataset = "CHES_2025"
model_type = "cnn"
leakage = "HW"
root = "./Result/"
save_root = root + dataset + "_" + model_type + "_" + leakage + "/"
model_root = save_root + "models/"
# Set this to the total number of models you trained
total_num_models = 20 # Or 100, etc.

# --- Setup for the QUICK results CSV log ---
log_file_path = os.path.join(save_root, "evaluation_log_QUICK.csv")
log_headers = ["model_index", "final_ge", "ntge", "time_taken_s"]
with open(log_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(log_headers)

# --- Load and preprocess the attack data ONCE ---
print("Loading attack data...")
data_root = 'Dataset/CHES_2025/CHES_Challenge.h5'
# Load the full attack set
(_, X_attack), (_, _), (_, plt_attack), correct_key = load_ctf_2025(
    './../' + data_root, leakage_model="HW", byte=0, train_end=1, test_end=100000
)

# --- THIS IS THE NEW, CRITICAL PART ---
# To align the attack data, we must use the same reference trace as in training.
# Load the first 1000 profiling traces just to create this reference.
(X_profiling_for_ref, _), _, _, _ = load_ctf_2025(
    './../' + data_root, leakage_model="HW", byte=0, train_end=1000, test_end=1
)
print("Creating reference trace for alignment...")
reference_trace = np.mean(X_profiling_for_ref, axis=0)
# This MUST be the same value you used in your main_pytorch.py script
max_shift = 50
# ---

num_sample_pts = X_attack.shape[1]

# Apply the EXACT SAME preprocessing pipeline as in training
print("Applying data preprocessing...")
print("Step 1: Alignment...")
X_attack = align_traces(X_attack, reference_trace, max_shift)
print("Step 2: Normalization...")
X_attack = normalize_trace(X_attack) # Note: I changed normalize_trace to normalize_trace
print("Preprocessing complete.")

# Define the leakage function
if leakage == 'HW':
    hw = np.array([bin(x).count("1") for x in range(256)])
    sbox = np.array(AES_Sbox)
    def leakage_fn(att_plt, k):
        sbox_in = np.bitwise_xor(att_plt.astype(np.uint8), k)
        sbox_out = sbox[sbox_in]
        return hw[sbox_out]
    classes = 9
else: # ID
    classes = 256
print("Data ready for quick evaluation.")

# --- Loop through all trained models and evaluate them QUICKLY ---
for i in range(total_num_models):
    print(f"\n--- Evaluating Model {i}/{total_num_models-1} (Quick Pass) ---")
    start_time = time.time()
    config_path = os.path.join(model_root, f"model_configuration_{i}.npy")
    model_path = os.path.join(model_root, f"model_{i}.pth")

    if not os.path.exists(model_path):
        print(f"Model {i} not found. Skipping.")
        continue

    try:
        config = np.load(config_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Could not load config for model {i}. Error: {e}. Skipping.")
        continue

    if model_type == "cnn":
        model = CNN(config, num_sample_pts, classes).to(device)
    else: # mlp
        model = MLP(config, num_sample_pts, classes).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run the evaluation with a small number of attacks for speed
    GE, NTGE = evaluate(device, model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn,
                        nb_attacks=10,  # <-- LOW NUMBER FOR SPEED
                        total_nb_traces_attacks=100000,
                        nb_traces_attacks=100000)

    final_ge = GE[-1]
    time_taken = time.time() - start_time
    print(f"Quick Result for Model {i}: Final GE = {final_ge:.2f}, NTGE = {NTGE}, Time = {time_taken:.2f}s")
    
    # You can still save the detailed .npy result if you want
    np.save(model_root + "/result_"+str(i), {"GE": GE, "NTGE": NTGE})
    
    # Log the summary result to the CSV
    with open(log_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([i, final_ge, NTGE, time_taken])

print(f"\n--- Quick evaluation complete. Analyze the results in {log_file_path} to find your champion models. ---")