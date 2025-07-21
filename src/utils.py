# utils.py
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from typing import Tuple

# AES S-box and its inverse
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
], dtype=np.uint8)

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
# Pre-computed Hamming Weight lookup table for efficiency
HW_TABLE = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
def HW(s):
    return bin(s).count("1")

def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[int(s)] for s in data]

def load_ctf_2025(filename, leakage_model='HW', byte=0, train_begin=0, train_end=100000, test_begin=0, test_end=100000, poi_start=0, poi_end=7000):
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

def NTGE_fn(ge_curve: np.ndarray) -> int:
    """
    Calculates the Number of Traces to Guessing Entropy zero (NTGE).

    This is the first trace index `t` where the GE curve hits 0 and stays there.
    If GE never reaches 0, it returns the total number of traces in the curve.

    Args:
        ge_curve (np.ndarray): The Guessing Entropy curve, where ge_curve[t] is the
                               average rank after t+1 traces.

    Returns:
        int: The NTGE value.
    """
    # Find the index of the last occurrence of a non-zero value
    non_zero_indices = np.where(ge_curve > 0)[0]
    if len(non_zero_indices) == 0:
        return 0  # GE is 0 from the first trace

    last_non_zero_index = non_zero_indices[-1]
    
    # NTGE is the next index. If the last non-zero value is the last element
    # of the curve, it means GE never stabilized at 0.
    ntge = last_non_zero_index + 1
    if ntge >= len(ge_curve):
      return len(ge_curve)
    
    return int(ntge)


def evaluate(
    device: torch.device,
    model: torch.nn.Module,
    X_attack: np.ndarray,
    plt_attack: np.ndarray,
    correct_key: int,
    nb_attacks: int = 100,
    nb_traces_attacks: int = 2000,
    batch_size: int = 1024
) -> Tuple[np.ndarray, int]:
    """
    Performs a vectorized side-channel attack to compute Guessing Entropy and NTGE.

    This function is a fast, memory-efficient, and mathematically equivalent
    reimplementation of the original `evaluate` function. It adheres to the
    CHES 2025 GE_WAR challenge rules.

    Args:
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        model (torch.nn.Module): The trained neural network model.
        X_attack (np.ndarray): The attack traces.
        plt_attack (np.ndarray): The plaintexts corresponding to the attack traces.
        correct_key (int): The correct key byte (0-255).
        nb_attacks (int): The number of attack experiments to average over (e.g., 100).
        nb_traces_attacks (int): The number of traces to use for the GE curve.
        batch_size (int): The batch size for the model's forward pass.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - GE_curve (np.ndarray): The final Guessing Entropy curve.
            - NTGE (int): The calculated NTGE.
    """
    model.eval()
    
    total_nb_traces = X_attack.shape[0]
    if nb_traces_attacks > total_nb_traces:
        raise ValueError(f"nb_traces_attacks ({nb_traces_attacks}) cannot be greater than the number of available attack traces ({total_nb_traces})")

    # ---------- 1. Forward pass in batches for memory efficiency ---------- #
    ds = TensorDataset(torch.from_numpy(X_attack))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    logp_chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device).unsqueeze(1).float()
            # Use autocast for potential performance gain on compatible GPUs
            with torch.amp.autocast(device_type=device.type):
                logits = model(batch)
            logp_chunks.append(F.log_softmax(logits, dim=1).cpu())
            
    # logp contains the log-probabilities for all attack traces
    logp = torch.cat(logp_chunks).numpy()  # Shape: (total_nb_traces, C)
    num_classes = logp.shape[1]            # Number of output classes (e.g., 9 for HW, 256 for ID)

    # ---------- 2. Prepare for parallel attacks ---------- #
    # This matrix will hold the running sum of log-probabilities for each key guess
    # across all attack experiments. Shape: (nb_attacks, 256)
    key_log_probs = np.zeros((nb_attacks, 256), dtype=np.float64)
    GE_curve = np.empty(nb_traces_attacks, dtype=np.float32)

    # Pre-generate random permutations for all attacks to ensure reproducibility
    # and to select traces randomly for each experiment.
    shuffles = [np.random.permutation(total_nb_traces) for _ in range(nb_attacks)]

    print(f"Starting {nb_attacks} attack simulations...")
    # ---------- 3. Vectorized rank computation over traces ---------- #
    for t in trange(nb_traces_attacks, desc="Calculating GE Curve"):
        # For each trace `t`, get the corresponding shuffled trace index for each attack run
        trace_indices = np.array([s[t] for s in shuffles])  # Shape: (nb_attacks,)

        # Get the log-probabilities and plaintexts for the selected traces
        logp_slice = logp[trace_indices]      # Shape: (nb_attacks, num_classes)
        pt_slice = plt_attack[trace_indices]  # Shape: (nb_attacks,)

        # Vectorized calculation of leakage values (z_i,k) for all key guesses
        # The result 'indices' will have shape (nb_attacks, 256)
        if num_classes == 256:  # Identity (ID) leakage model
            sbox_out = AES_Sbox[pt_slice[:, None] ^ np.arange(256, dtype=np.uint8)]
            indices = sbox_out
        elif num_classes == 9: # Hamming Weight (HW) leakage model
            sbox_out = AES_Sbox[pt_slice[:, None] ^ np.arange(256, dtype=np.uint8)]
            indices = HW_TABLE[sbox_out]
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}. Only 9 (HW) and 256 (ID) are supported.")
        
        # `take_along_axis` efficiently gathers the log-probabilities corresponding
        # to the calculated leakage for each key guess.
        aligned_logp = np.take_along_axis(logp_slice, indices, axis=1)

        # Update the total log-probability for each key guess in each attack
        key_log_probs += aligned_logp

        # Calculate the rank of the correct key for all attacks simultaneously
        ranks = np.argsort(np.argsort(-key_log_probs, axis=1), axis=1)
        
        # The GE for trace `t` is the average rank of the correct key across all attacks
        GE_curve[t] = ranks[:, correct_key].mean()

    # ---------- 4. Final Metrics ---------- #
    NTGE = NTGE_fn(GE_curve)
    
    final_ge = GE_curve[-1]
    print("\n--- Evaluation Complete ---")
    print(f"Final GE: {final_ge:.2f}")
    print(f"NTGE: {NTGE}")
    
    return GE_curve, NTGE

