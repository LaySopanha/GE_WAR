import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_ctf_2025 # We'll keep the loader in utils

class DataAugmentation:
    """
    Applies random horizontal shift and adds Gaussian noise to a trace.
    """
    def __init__(self, max_shift, noise_level):
        self.max_shift = max_shift
        self.noise_level = noise_level

    def __call__(self, sample):
        trace, sensitive = sample['trace'], sample['sensitive']
        
        # Random shift
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        trace = np.roll(trace, shift, axis=-1)
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_level, trace.shape).astype(np.float32)
        trace += noise
        
        sample['trace'] = trace
        return sample

class ToTensor_trace:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        trace, label = sample['trace'], sample['sensitive']
        return torch.from_numpy(trace).float(), torch.from_numpy(np.array(label)).long()

class Custom_Dataset(Dataset):
    def __init__(self, root='./', dataset="CHES_2025", leakage="ID", poi_start=0, poi_end=7000, 
                 train_end=500000, test_end=100000, transform=None):
        
        data_root = os.path.join(root, 'Dataset/CHES_2025/CHES_Challenge.h5')
        print(f"Loading dataset: {data_root}")
        
        (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), \
        (self.plt_profiling, self.plt_attack), self.correct_key = \
            load_ctf_2025(data_root, leakage_model=leakage, byte=0, 
                          train_end=train_end, test_end=test_end)

        print(f"Applying POI: Slicing traces from {poi_start} to {poi_end}.")
        self.X_profiling = self.X_profiling[:, poi_start:poi_end]
        self.X_attack = self.X_attack[:, poi_start:poi_end]

        print("Standardizing profiling data and fitting scaler...")
        self.scaler = StandardScaler()
        self.X_profiling = self.scaler.fit_transform(self.X_profiling)
        
        # Reshape to (num_traces, 1, num_points) for PyTorch Conv1d
        self.X_profiling = np.expand_dims(self.X_profiling, 1)
        self.X_attack = np.expand_dims(self.X_attack, 1)
        
        self.transform = transform
        self.phase = "train" # Default phase

    def split_attack_set_validation_test(self, validation_size=0.1):
        # The attack set is split into validation (for hyperparameter tuning) and a final test set (unused during tuning)
        self.X_attack_val, self.X_attack_test, \
        self.Y_attack_val, self.Y_attack_test, \
        self.plt_attack_val, self.plt_attack_test = \
            train_test_split(self.X_attack, self.Y_attack, self.plt_attack, 
                             test_size=1-validation_size, random_state=42, stratify=self.Y_attack)
        print(f"Attack set split: {len(self.X_attack_val)} for validation, {len(self.X_attack_test)} for final test.")

    def choose_phase(self, phase):
        self.phase = phase
        if phase == 'train':
            self.X, self.Y = self.X_profiling, self.Y_profiling
        elif phase == 'validation':
            self.X, self.Y, self.Plaintext = self.X_attack_val, self.Y_attack_val, self.plt_attack_val
        elif phase == 'test':
            self.X, self.Y, self.Plaintext = self.X_attack_test, self.Y_attack_test, self.plt_attack_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        trace = self.X[idx]
        sensitive = self.Y[idx]
        sample = {'trace': trace, 'sensitive': sensitive}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample