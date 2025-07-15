# src/dataloader.py

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import load_ctf_2025
import torch

class DataAugmentation(object):
    """Applies random augmentations to side-channel traces."""
    def __init__(self, shift_prob=0.5, noise_prob=0.5, max_shift=10, noise_level=0.01):
        self.shift_prob = shift_prob
        self.noise_prob = noise_prob
        self.max_shift = max_shift
        self.noise_level = noise_level

    def __call__(self, sample):
        trace = sample['trace']
        if np.random.rand() < self.shift_prob:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            trace = np.roll(trace, shift, axis=-1)
        if np.random.rand() < self.noise_prob:
            noise = np.random.normal(0, self.noise_level, trace.shape).astype(trace.dtype)
            trace += noise
        sample['trace'] = trace
        return sample

class Custom_Dataset(Dataset):
    def __init__(self, root = './', dataset = "CHES_2025", leakage = "HW",transform = None,
                 poi_start=0, poi_end=7000, train_end=45000, test_end=10000):
        if dataset == "CHES_2025":
            byte = 0
            data_root = 'Dataset/CHES_2025/CHES_Challenge.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
                self.plt_profiling, self.plt_attack), self.correct_key = load_ctf_2025(
                root + data_root, leakage_model=leakage, byte=byte, 
                train_begin=0, train_end=train_end, test_begin=0, test_end=test_end,
                poi_start=poi_start, poi_end=poi_end)

        print("The dataset we are using: ", data_root)
        self.transform = transform
        self.scaler_std = StandardScaler()
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)

    def split_attack_set_validation_test(self):
        (self.X_attack_test, self.X_attack_val, 
         self.Y_attack_test, self.Y_attack_val,
         self.plt_attack_test, self.plt_attack_val) = train_test_split(
            self.X_attack, self.Y_attack, self.plt_attack, test_size=0.1, random_state=0)

    def choose_phase(self,phase):
        if phase == 'train':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_profiling, 1), self.Y_profiling, self.plt_profiling
        elif phase == 'validation':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_attack_val, 1), self.Y_attack_val, self.plt_attack_val
        elif phase == 'test':
            self.X, self.Y, self.Plaintext = np.expand_dims(self.X_attack_test, 1), self.Y_attack_test, self.plt_attack_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        sample = {'trace': trace, 'sensitive': sensitive}

        if self.transform:
            sample = self.transform(sample)

        return sample['trace'], sample['sensitive']

class ToTensor_trace(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        trace, label = sample['trace'], sample['sensitive']
        return {'trace': torch.from_numpy(trace).float(), 
                'sensitive': torch.from_numpy(np.array(label)).long()}
