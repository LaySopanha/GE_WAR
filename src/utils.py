import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

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

def load_ctf_2025(filename, leakage_model='ID', byte=0, train_end=500000, test_end=100000):
    in_file = h5py.File(filename, "r")
    # Profiling data
    X_profiling = np.array(in_file['Profiling_traces/traces'][:train_end], dtype=np.float32)
    plt_profiling = np.array(in_file['Profiling_traces/metadata'][:train_end]['plaintext'][:, byte])
    Y_profiling = np.array(in_file['Profiling_traces/metadata'][:train_end]['labels'], dtype=np.uint8) # ID labels are byte 0

    # Attack data
    X_attack = np.array(in_file['Attack_traces/traces'][:test_end], dtype=np.float32)
    plt_attack = np.array(in_file['Attack_traces/metadata'][:test_end]['plaintext'][:, byte])
    Y_attack = np.array(in_file['Attack_traces/metadata'][:test_end]['labels'], dtype=np.uint8)
    
    correct_key = np.array(in_file['Attack_traces/metadata'][0]['key'][byte])
    in_file.close()
    
    # NOTE: The provided labels are for the ID model (S-box output). If a HW model were needed,
    # it would be calculated from these ID labels. The current sweep uses ID, so no conversion is needed.
    
    print(f"Profiling traces: {X_profiling.shape}, Attack traces: {X_attack.shape}")
    print(f"Correct key for byte {byte} is: {correct_key}")
    
    return (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key

def rk_key(rank_array, key):
    key_val = rank_array[key]
    sorted_rank = np.sort(rank_array)[::-1]
    return np.where(sorted_rank == key_val)[0][0]

def rank_compute(prediction, att_plt, correct_key, leakage_fn):
    (nb_traces, nb_hyp) = prediction.shape
    key_log_prob = np.zeros(256)
    prediction = np.log(prediction + 1e-40)
    rank_evol = np.full(nb_traces, 255, dtype=np.int16)

    for i in range(nb_traces):
        for k in range(256):
            y_value = leakage_fn(att_plt[i], k)
            key_log_prob[k] += prediction[i, y_value]
        rank_evol[i] = rk_key(key_log_prob, correct_key)
    return rank_evol

def perform_attacks(max_traces, predictions, plt_attack, correct_key, leakage_fn, nb_attacks=100):
    all_rk_evol = np.zeros((nb_attacks, max_traces), dtype=np.int16)
    
    for i in tqdm(range(nb_attacks), desc="Performing attacks"):
        # Shuffle data for each attack run
        indices = np.arange(len(predictions))
        random.shuffle(indices)
        shuffled_predictions = predictions[indices]
        shuffled_plt = plt_attack[indices]
        
        # Calculate rank evolution for this run
        all_rk_evol[i] = rank_compute(shuffled_predictions[:max_traces], shuffled_plt[:max_traces], correct_key, leakage_fn)
        
    return np.mean(all_rk_evol, axis=0)

def NTGE_fn(GE):
    if GE is None: return float('inf')
    ntge = float('inf')
    for i in range(len(GE) - 1, -1, -1):
        if GE[i] > 0:
            break
        elif GE[i] == 0:
            ntge = i + 1 # Number of traces is index + 1
    return ntge

def evaluate(device, model, X_attack, plt_attack, correct_key, leakage_fn, nb_attacks=50, max_traces=10000):
    model.eval()
    if len(X_attack) == 0:
        print("Warning: No attack traces provided for evaluation.")
        return [], float('inf'), 255 # Return a default bad GE

    # Ensure max_traces does not exceed available traces
    max_traces = min(max_traces, len(X_attack))
    print(f"Evaluating GE/NTGE with {max_traces} traces over {nb_attacks} attacks...")

    # Predict probabilities in batches to avoid OOM errors
    batch_size = 512
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(X_attack), batch_size):
            batch_x = torch.from_numpy(X_attack[i:i+batch_size]).to(device).float()
            preds = F.softmax(model(batch_x), dim=1).cpu().numpy()
            all_predictions.append(preds)
    
    predictions = np.concatenate(all_predictions)

    # Perform attacks and get average GE curve
    GE = perform_attacks(max_traces, predictions, plt_attack, correct_key, leakage_fn, nb_attacks)
    NTGE = NTGE_fn(GE)
    
    ### --- NEW --- ###
    # Get the final GE value at the maximum number of traces.
    # If the GE curve is empty for some reason, default to the worst possible rank (255).
    final_ge_at_max_traces = GE[-1] if (GE is not None and len(GE) > 0) else 255
    ### --- END NEW --- ###
    
    print(f"Final GE after {max_traces} traces: {final_ge_at_max_traces:.2f}")
    print(f"Final NTGE: {NTGE}")
    
    ### --- CHANGED --- ###
    # Return all three metrics now
    return GE, NTGE, final_ge_at_max_traces