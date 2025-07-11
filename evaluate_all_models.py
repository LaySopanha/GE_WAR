# evaluate_champions_FINAL.py

import numpy as np
import os
import torch
import csv
import time
from src.net import MLP, CNN
from src.utils import AES_Sbox, calculate_HW, normalize_traces, evaluate
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

# --- KEY CHANGE: Manually list your best models from the quick pass ---
# After analyzing 'evaluation_log_QUICK.csv', put the best model indices here.
champion_indices = [78, 42, 15] # <-- EXAMPLE: REPLACE WITH YOURS
if not champion_indices:
    print("Please specify the champion model indices to evaluate.")
    exit()
# ---

# --- Setup for the FINAL results CSV log ---
log_file_path = os.path.join(save_root, "evaluation_log_FINAL.csv")
log_headers = ["model_index", "final_ge", "ntge", "time_taken_s"]
with open(log_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(log_headers)

# --- Load and preprocess the attack data ONCE (same as before) ---
print("Loading attack data...")
data_root = 'Dataset/CHES_2025/CHES_Challenge.h5'
(_, X_attack), (_, _), (_, plt_attack), correct_key = load_ctf_2025(
    './../' + data_root, leakage_model="HW", byte=0, train_end=1, test_end=100000
)
num_sample_pts = X_attack.shape[1]
print("Applying data preprocessing...")
X_attack = normalize_traces(X_attack)
# If you add alignment, add it here:
# X_attack = align_traces(X_attack)
print("Preprocessing complete.")

# Define leakage function (same as before)
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
print("Data ready for final evaluation.")

# --- Loop through ONLY the champion models and evaluate them FULLY ---
for i in champion_indices: # <-- This now loops over your short list
    print(f"\n--- Evaluating Champion Model {i} (Full Pass) ---")
    start_time = time.time()
    config_path = os.path.join(model_root, f"model_configuration_{i}.npy")
    model_path = os.path.join(model_root, f"model_{i}.pth")

    if not os.path.exists(model_path):
        print(f"Champion model {i} not found. Skipping.")
        continue

    try:
        config = np.load(config_path, allow_pickle=True).item()
    except:
        print(f"Could not load config for champion model {i}. Skipping.")
        continue

    if model_type == "cnn":
        model = CNN(config, num_sample_pts, classes).to(device)
    else: # mlp
        model = MLP(config, num_sample_pts, classes).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Run the FULL, slow evaluation with the official number of attacks ---
    GE, NTGE = evaluate(device, model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn,
                        nb_attacks=100, # <-- OFFICIAL VALUE (100)
                        total_nb_traces_attacks=100000,
                        nb_traces_attacks=100000)

    final_ge = GE[-1]
    time_taken = time.time() - start_time
    print(f"Final Result for Model {i}: Final GE = {final_ge:.2f}, NTGE = {NTGE}, Time = {time_taken:.2f}s")
    np.save(model_root + "/result_"+str(i), {"GE": GE, "NTGE": NTGE})
    # Log the final, high-quality result to the CSV
    with open(log_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([i, final_ge, NTGE, time_taken])

print(f"\n--- Final evaluation of champions complete. Official scores are in {log_file_path} ---")