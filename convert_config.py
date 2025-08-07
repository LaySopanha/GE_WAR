#!/usr/bin/env python3
"""
Convert YAML config to NPY config for training
"""
import yaml
import numpy as np
import sys

def convert_yaml_to_npy(yaml_file, npy_file):
    """Convert YAML config to NPY config format"""
    
    # Load YAML config
    with open(yaml_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Extract config values from the wandb format
    config_dict = {}
    
    # Navigate through the wandb config structure
    if '_wandb' in yaml_config and 'value' in yaml_config['_wandb']:
        # This is a wandb config file - extract from the nested structure
        for key, value_dict in yaml_config.items():
            if key != '_wandb' and isinstance(value_dict, dict) and 'value' in value_dict:
                config_dict[key] = value_dict['value']
    else:
        # Direct config format
        config_dict = yaml_config
    
    # Ensure we have the essential parameters
    required_params = {
        'activation': 'selu',
        'attack_eval_frequency': 5,
        'batch_size': 64,
        'conv_layers': 2,
        'dataset': 'CHES_2025',
        'dropout_rate': 0.1659671608310503,
        'early_stopping_patience': 10,
        'epochs': 100,
        'filters': 16,
        'k_folds': 2,
        'kernels': 36,
        'layers': 3,
        'leakage': 'ID',
        'lr': 0.0005,
        'max_shift': 10,
        'min_epochs': 30,
        'model_type': 'cnn',
        'neurons': 128,
        'noise_level': 0.04286598024507582,
        'num_poi': 100,
        'num_traces_attack': 100000,
        'optimizer': 'Adam',
        'padding': 1,
        'poi_end': 7000,
        'poi_start': 0,
        'pooling_sizes': 4,
        'pooling_types': 'max_pool',
        'train_end': 500000  # Add default train_end
    }
    
    # Use extracted values or defaults
    final_config = {}
    for key, default_value in required_params.items():
        final_config[key] = config_dict.get(key, default_value)
    
    # Save as NPY file
    np.save(npy_file, final_config)
    
    print(f"✅ Converted config from {yaml_file} to {npy_file}")
    print(f"📋 Config parameters:")
    for key, value in final_config.items():
        print(f"  {key}: {value}")
    
    return final_config

if __name__ == "__main__":
    yaml_file = "config-ntge_86825.yaml"
    npy_file = "config-ntge_86825.npy"
    
    config = convert_yaml_to_npy(yaml_file, npy_file)
