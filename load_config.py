import numpy as np
import json

# Load best model config
try:
    config = np.load('ge0_config_run_1jcrispy_fold_0_ntge_97397.npy', allow_pickle=True).item()
    print("🎯 BEST MODEL CONFIG (NTGE=97397):")
    print("="*50)
    for k, v in config.items():
        print(f"{k}: {v}")
except Exception as e:
    print(f"Error loading config: {e}")
