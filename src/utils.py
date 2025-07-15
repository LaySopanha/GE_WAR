# src/utils.py

import math
import random
import h5py
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])
# Other arrays like AES_Sbox_inv are omitted for brevity but should be kept if they exist in your original file

def HW(s):
    return bin(s).count("1")

def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[int(s)] for s in data]

def load_ctf_2025(filename, leakage_model='HW', byte = 0, train_begin = 0, train_end = 100000,test_begin = 0, test_end = 50000,
                  poi_start=0, poi_end=7000):
    in_file = h5py.File(filename, "r")
    
    print(f"Loading traces from slice [{poi_start}:{poi_end}]...")
    X_profiling = np.array(in_file['Profiling_traces/traces'][:, poi_start:poi_end])
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))

    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][:, byte])
    if byte != 0:
        key_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'][:,byte])
        Y_profiling = np.zeros(P_profiling.shape[0])
        for i in range(len(P_profiling)):
            Y_profiling[i] = AES_Sbox[P_profiling[i] ^ key_profiling[i]]
        if leakage_model == 'HW': Y_profiling = calculate_HW(Y_profiling)
    else:
        Y_profiling = np.array(in_file['Profiling_traces/metadata'][:]['labels'])
        if leakage_model == 'HW': Y_profiling = calculate_HW(Y_profiling)

    X_attack = np.array(in_file['Attack_traces/traces'][:, poi_start:poi_end])
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][:, byte])
    attack_key = np.array(in_file['Attack_traces/metadata'][:]['key'][0, byte])
    
    if byte != 0:
        key_attack = np.array(in_file['Attack_traces/metadata'][:]['key'][:,byte])
        Y_attack = np.zeros(P_attack.shape[0])
        for i in range(len(P_attack)):
            Y_attack[i] = AES_Sbox[P_attack[i] ^ key_attack[i]]
        if leakage_model == 'HW': Y_attack = calculate_HW(Y_attack)
    else:
        Y_attack = np.array(in_file['Attack_traces/metadata'][:]['labels'])
        if leakage_model == 'HW': Y_attack = calculate_HW(Y_attack)

    return (X_profiling[train_begin:train_end], X_attack[test_begin:test_end]), (Y_profiling[train_begin:train_end],  Y_attack[test_begin:test_end]),\
           (P_profiling[train_begin:train_end],  P_attack[test_begin:test_end]), attack_key

def rk_key(rank_array, key):
    try:
        key_val = rank_array[key]
        final_rank = np.where(np.sort(rank_array)[::-1] == key_val)[0][0]
    except (ValueError, IndexError):
        final_rank = 256
    
    return float(final_rank) if not (math.isnan(float(final_rank)) or math.isinf(float(final_rank))) else 256.0

def rank_compute(prediction, att_plt, correct_key,leakage_fn):
    (nb_traces, _) = prediction.shape
    key_log_prob = np.zeros(256)
    prediction = np.log(prediction + 1e-40)
    rank_evol = np.full(nb_traces, 255, dtype=np.float32)
    for i in range(nb_traces):
        for k in range(256):
            y_value = leakage_fn(att_plt[i], k)
            key_log_prob[k] += prediction[i,  y_value]
        rank_evol[i] =  rk_key(key_log_prob, correct_key)
    return rank_evol, key_log_prob

def perform_attacks(nb_traces, predictions, plt_attack,correct_key,leakage_fn,nb_attacks=1, shuffle=True):
    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    for i in tqdm(range(nb_attacks), desc="Attack Progress"):
        indices = np.arange(len(predictions))
        if shuffle: np.random.shuffle(indices)
        shuffled_predictions = predictions[indices]
        shuffled_plt = plt_attack[indices]
        rank_evol, _ = rank_compute(shuffled_predictions[:nb_traces], shuffled_plt[:nb_traces], correct_key, leakage_fn=leakage_fn)
        all_rk_evol[i] = rank_evol
    return np.mean(all_rk_evol, axis=0), None

# In src/utils.py

def NTGE_fn(GE):
    # Find the last index where the rank is not 0
    # If all ranks are 0, non_zero_indices will be empty
    non_zero_indices = np.where(GE > 0)[0]
    
    if len(non_zero_indices) == 0:
        # If the GE is 0 from the very first trace
        return 1
    else:
        # The first trace where GE is stably 0 is one after the last non-zero rank
        last_non_zero_idx = non_zero_indices[-1]
        if last_non_zero_idx + 1 >= len(GE):
            # This means the GE never stayed at 0 until the end
            return float('inf')
        else:
            return last_non_zero_idx + 2 # +1 for index, +1 for next trace

def evaluate(device, model, X_attack, plt_attack,correct_key,leakage_fn, nb_attacks=100, total_nb_traces_attacks=2000, nb_traces_attacks = 1700):
    model.eval()
    with torch.no_grad():
        attack_traces_tensor = torch.from_numpy(X_attack[:total_nb_traces_attacks]).to(device).unsqueeze(1).float()
        predictions = F.softmax(model(attack_traces_tensor), dim=1).cpu().numpy()
    GE, _ = perform_attacks(nb_traces_attacks, predictions, plt_attack, correct_key, nb_attacks=nb_attacks, shuffle=True, leakage_fn=leakage_fn)
    NTGE = NTGE_fn(GE)
    print("\n--- Evaluation Results ---")
    print(f"Final GE after {nb_traces_attacks} traces: {GE[-1] if len(GE) > 0 else 'N/A'}")
    print(f"NTGE (first trace count where GE=0): {NTGE}")
    print("--------------------------\n")
    return GE, NTGE