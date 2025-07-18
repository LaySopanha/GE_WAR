# src/utils.py

import math, random, h5py, numpy as np, torch, wandb
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
def HW(s): return bin(s).count("1")
def calculate_HW(data): return [HW(int(s)) for s in data]

def calculate_snr(traces, labels):
    num_classes = len(np.unique(labels))
    num_samples = traces.shape[1]
    mean_per_class = np.zeros((num_classes, num_samples))
    var_per_class = np.zeros((num_classes, num_samples))

    for c in range(num_classes):
        class_traces = traces[labels == c]
        if len(class_traces) > 0:
            mean_per_class[c] = np.mean(class_traces, axis=0)
            var_per_class[c] = np.var(class_traces, axis=0)

    signal_variance = np.var(mean_per_class, axis=0)
    noise_variance = np.mean(var_per_class, axis=0)
    
    # To avoid division by zero
    snr = signal_variance / (noise_variance + 1e-10)
    return snr

def load_ctf_2025(filename, leakage_model='HW', byte=0, train_begin=0, train_end=100000, test_begin=0, test_end=50000, poi_start=0, poi_end=7000):
    with h5py.File(filename, "r") as in_file:
        print(f"Loading traces from slice [{poi_start}:{poi_end}]...")
        X_profiling = np.array(in_file['Profiling_traces/traces'][train_begin:train_end, poi_start:poi_end], dtype=np.float32)
        P_profiling = np.array(in_file['Profiling_traces/metadata'][train_begin:train_end]['plaintext'][:, byte])
        Y_profiling = np.array(in_file['Profiling_traces/metadata'][train_begin:train_end]['labels'])
        if leakage_model == 'HW': Y_profiling = calculate_HW(Y_profiling)

        X_attack = np.array(in_file['Attack_traces/traces'][test_begin:test_end, poi_start:poi_end], dtype=np.float32)
        P_attack = np.array(in_file['Attack_traces/metadata'][test_begin:test_end]['plaintext'][:, byte])
        Y_attack = np.array(in_file['Attack_traces/metadata'][test_begin:test_end]['labels'])
        if leakage_model == 'HW': Y_attack = calculate_HW(Y_attack)
        
        attack_key = np.array(in_file['Attack_traces/metadata'][0]['key'][byte])
    return (X_profiling, X_attack), (Y_profiling, Y_attack), (P_profiling, P_attack), attack_key

def rk_key(rank_array, key):
    try:
        key_val = rank_array[key]
        sorted_ranks = np.sort(rank_array)[::-1]
        final_rank = np.where(sorted_ranks == key_val)[0][0]
    except (ValueError, IndexError): final_rank = 255.0
    return float(final_rank)

def rank_compute_vectorized(predictions, att_plt, correct_key, leakage_fn):
    nb_traces = predictions.shape[0]
    key_guesses = np.arange(256)
    plaintext_matrix = np.tile(att_plt, (256, 1)).transpose()
    sbox_out = AES_Sbox[plaintext_matrix ^ key_guesses]
    if leakage_fn.__name__ == '<lambda>': # A trick to detect which leakage model
        leakage_matrix = sbox_out
    else:
        hw_lut = np.array([HW(s) for s in range(256)]); leakage_matrix = hw_lut[sbox_out]
    log_preds = np.log(predictions + 1e-40)
    trace_indices = np.arange(nb_traces)
    indexed_log_probs = log_preds[trace_indices[:, None], leakage_matrix]
    key_log_prob_evolution = np.cumsum(indexed_log_probs, axis=0)
    rank_evol = np.array([rk_key(key_log_prob_evolution[i], correct_key) for i in range(nb_traces)])
    return rank_evol, key_log_prob_evolution[-1]

def perform_attacks(nb_traces, predictions, plt_attack, correct_key, leakage_fn, nb_attacks=1, shuffle=True):
    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    for i in range(nb_attacks):
        indices = np.arange(len(predictions))
        if shuffle: np.random.shuffle(indices)
        shuffled_predictions, shuffled_plt = predictions[indices], plt_attack[indices]
        rank_evol, _ = rank_compute_vectorized(shuffled_predictions[:nb_traces], shuffled_plt[:nb_traces], correct_key, leakage_fn=leakage_fn)
        all_rk_evol[i] = rank_evol
    return np.mean(all_rk_evol, axis=0), None

def NTGE_fn(GE):
    non_zero_indices = np.where(GE > 0)[0]
    if len(non_zero_indices) == 0: return 1
    last_non_zero_idx = non_zero_indices[-1]
    return float('inf') if last_non_zero_idx + 1 >= len(GE) else last_non_zero_idx + 2


HW_LUT = np.array([bin(x).count("1") for x in range(256)]) # Pre-compute HW lookup table

def evaluate(device, model,
             X_attack, plt_attack, correct_key,
             leakage_fn,
             nb_attacks              = 100,
             total_nb_traces_attacks = 2000,
             nb_traces_attacks       = 1700,
             batch_size              = 512):

    model.eval()

    # ---------- 1. forward pass in batches ------------------------------- #
    ds     = TensorDataset(torch.from_numpy(X_attack[:total_nb_traces_attacks]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logp_chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            batch  = batch.to(device).float()
            # The model expects a 3D tensor (batch, channels, length)
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(1)
            
            # Use mixed precision for speed on compatible GPUs
            with torch.amp.autocast(device_type=str(device).split(':')[0]):
                logits = model(batch)

            logp_chunks.append(F.log_softmax(logits, dim=1).cpu())
            
    logp = torch.cat(logp_chunks).numpy()
    C    = logp.shape[1]

    # ---------- 2. prep containers -------------------------------------- #
    key_probs_runs = np.zeros((nb_attacks, 256), dtype=np.float64)
    GE_curve       = np.empty(nb_traces_attacks, dtype=np.float32)

    # Pre-generate all random indices at once for speed
    shuffles = [np.random.permutation(total_nb_traces_attacks) for _ in range(nb_attacks)]

    # ---------- 3. vectorised rank update -------------------------------- #
    key_range = np.arange(256)
    for t in range(nb_traces_attacks):
        # Get the indices for the current trace number across all attack runs
        idxs     = np.array([s[t] for s in shuffles])
        lp_slice = logp[idxs]
        pt_slice = plt_attack[idxs]

        # Vectorized leakage calculation
        if C == 256: # ID leakage
            indices = AES_Sbox[pt_slice[:, None] ^ key_range]
        else: # HW leakage (C==9)
            xor_result = AES_Sbox[pt_slice[:, None] ^ key_range]
            indices = HW_LUT[xor_result]

        # Gather the log-probabilities for the hypothetical leakages
        aligned = np.take_along_axis(lp_slice, indices, axis=1)
        key_probs_runs += aligned

        # Get the rank of the correct key for all attack runs
        ranks = np.argsort(np.argsort(-key_probs_runs, axis=1), axis=1)
        # Average the rank of the correct key to get the GE for this trace number
    GE_curve[t] = ranks[:, correct_key].mean()

    # Create and log a detailed table of key ranks
    final_key_log_probs = key_probs_runs.mean(axis=0)
    final_ranks = np.argsort(np.argsort(-final_key_log_probs))
    table = wandb.Table(columns=["Key (Hex)", "Final Log-Prob", "Rank"])
    for k in range(256):
        key_hex = f"0x{k:02x}"
        table.add_data(key_hex, final_key_log_probs[k], final_ranks[k])
    wandb.log({"key_rank_distribution": table})

    NTGE = NTGE_fn(GE_curve)
    final_ge = GE_curve[-1]
    print(f"\n--- Evaluation Results ---\nFinal GE: {final_ge:.2f} | NTGE: {NTGE}\n--------------------------\n")

    return GE_curve, NTGE, final_ge
