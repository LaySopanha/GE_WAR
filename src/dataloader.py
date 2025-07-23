# src/dataloader.py

import os, numpy as np, torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import load_ctf_2025

class DataAugmentation:
    def __init__(self, max_shift=15, noise_level=0.05, flip_prob=0.5):
        self.max_shift = max_shift
        self.noise_level = noise_level
        self.flip_prob = flip_prob

    def __call__(self, sample):
        trace = sample['trace']
        
        # Apply random horizontal flipping
        if np.random.rand() < self.flip_prob:
            trace = np.flip(trace, axis=0)

        # Apply random shift
        if self.max_shift > 0 and np.random.rand() < 0.7:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            trace = np.roll(trace, shift)
            
        # Apply random noise
        if self.noise_level > 0 and np.random.rand() < 0.7:
            noise = np.random.normal(0, self.noise_level, trace.shape).astype(trace.dtype)
            trace += noise
            
        sample['trace'] = trace
        return sample

class ToTensor_trace:
    def __call__(self, sample):
        trace, label = sample['trace'], sample['sensitive']
        return torch.from_numpy(trace.copy()).unsqueeze(0).float(), torch.from_numpy(np.array([label])).long()

class Custom_Dataset(Dataset):
    def __init__(self, root='./', dataset="CHES_2025", leakage="HW", transform=None,
                 poi_start=0, poi_end=7000, train_end=500000, test_end=100000):
        (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), \
        (self.plt_profiling, self.plt_attack), self.correct_key = load_ctf_2025(
            root + 'Dataset/CHES_2025/CHES_Challenge.h5', leakage_model=leakage, train_end=train_end,
            test_end=test_end, poi_start=poi_start, poi_end=poi_end)
        self.transform = transform
        # The scaler is now handled in the main script after POI selection.
        # self.scaler = StandardScaler()
        # if len(self.X_profiling) > 0: self.X_profiling = self.scaler.fit_transform(self.X_profiling)
        # if len(self.X_attack) > 0: self.X_attack = self.scaler.transform(self.X_attack)

    def split_attack_set_validation_test(self):
        (self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val,
         self.plt_attack_test, self.plt_attack_val) = train_test_split(
            self.X_attack, self.Y_attack, self.plt_attack, test_size=0.1, random_state=0)

    def choose_phase(self, phase):
        if phase == 'train': self.X, self.Y, self.Plaintext = self.X_profiling, self.Y_profiling, self.plt_profiling
        elif phase == 'validation': self.X, self.Y, self.Plaintext = self.X_attack_val, self.Y_attack_val, self.plt_attack_val
        elif phase == 'test': self.X, self.Y, self.Plaintext = self.X_attack_test, self.Y_attack_test, self.plt_attack_test

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        sample = {'trace': self.X[idx], 'sensitive': self.Y[idx]}
        if self.transform:
            return self.transform(sample)
        return sample['trace'], sample['sensitive']
