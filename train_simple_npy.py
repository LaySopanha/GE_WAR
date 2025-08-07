#!/usr/bin/env python3
"""
Simple NPY config training - stays close to main_pytorch.py approach
"""
import os, random, numpy as np, torch, wandb
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
from src.trainer import training_loop, attack_driven_training_loop
from src.utils import evaluate, AES_Sbox, calculate_snr

def load_npy_config(config_file):
    """Load config from NPY file"""
    config_dict = np.load(config_file, allow_pickle=True).item()
    print(f"📋 Loaded config from {config_file}:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    return config_dict

def run_experiment_with_npy_config(config_file):
    """Run experiment using NPY config file - SIMPLE VERSION like main_pytorch.py"""
    
    # Load config from NPY file
    config_dict = load_npy_config(config_file)
    
    # Initialize wandb with the loaded config
    run = wandb.init(config=config_dict, project="ge-war-ches2025")
    config = wandb.config
    
    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load full traces first to compute SNR
    train_end = getattr(config, 'train_end', 500000)  # Use config value or default to 500k
    full_dataset_obj = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage,
                                      poi_start=0, poi_end=7000,  # Load all points
                                      train_end=train_end, test_end=100000)
    
    print(f"🚀 Loading {train_end} profiling traces from dataset")

    # Calculate SNR on the full profiling dataset for accuracy - SAME AS MAIN
    snr = calculate_snr(full_dataset_obj.X_profiling, full_dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr)[-config.num_poi:]  # Use EXACT same POI count

    # Create a new dataset object with only the selected POIs - SAME AS MAIN
    dataset_obj = deepcopy(full_dataset_obj)
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]

    # Use the full profiling dataset (500k) instead of random subset - SAME AS MAIN
    print(f"📈 Using full {len(dataset_obj.X_profiling)} profiling traces (vs 200k subset)")

    # Handle single split vs k-fold cross-validation - SAME AS MAIN
    if config.k_folds == 1:
        # Single train/validation split (80/20)
        from sklearn.model_selection import train_test_split
        train_index, val_index = train_test_split(
            range(len(dataset_obj.X_profiling)), 
            test_size=0.2, 
            stratify=dataset_obj.Y_profiling, 
            random_state=SEED
        )
        fold_splits = [(train_index, val_index)]
        print(f"📊 Using single 80/20 train/validation split")
    else:
        # K-Fold Cross-Validation
        kf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=SEED)
        fold_splits = list(kf.split(dataset_obj.X_profiling, dataset_obj.Y_profiling))
        print(f"📊 Using {config.k_folds}-fold cross-validation")
    
    fold_results_ntge = []
    fold_results_ge = []
    fold_results_final_ge = []

    for fold, (train_index, val_index) in enumerate(fold_splits):
        if config.k_folds == 1:
            print(f"--- Single Train/Validation Split ---")
        else:
            print(f"--- Fold {fold+1}/{config.k_folds} ---")

        # Create a scaler for this fold - SAME AS MAIN
        scaler = StandardScaler()
        
        # Create datasets for this fold - SAME AS MAIN
        train_transform = transforms.Compose([
            DataAugmentation(max_shift=config.max_shift, noise_level=config.noise_level),
            ToTensor_trace()])
        eval_transform = transforms.Compose([ToTensor_trace()])

        # Create a deepcopy of the dataset object for this fold - SAME AS MAIN
        fold_dataset_obj = deepcopy(dataset_obj)
        
        # Get the training and validation data for this fold - SAME AS MAIN
        X_train, X_val = fold_dataset_obj.X_profiling[train_index], fold_dataset_obj.X_profiling[val_index]
        Y_train, Y_val = fold_dataset_obj.Y_profiling[train_index], fold_dataset_obj.Y_profiling[val_index]

        # Fit the scaler on the training data for this fold - SAME AS MAIN
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Create the dataloaders for this fold - SAME AS MAIN
        train_dataset = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage)
        train_dataset.X_profiling, train_dataset.Y_profiling = X_train, Y_train
        train_dataset.transform = train_transform
        train_dataset.choose_phase("train")
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

        val_dataset = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage)
        val_dataset.X_profiling, val_dataset.Y_profiling = X_val, Y_val
        val_dataset.transform = eval_transform
        val_dataset.choose_phase("train") # Use train phase to get profiling data
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        # Initialize model for this fold - SAME AS MAIN
        poi_width = config.num_poi  # Use ORIGINAL POI count
        classes = 256 if config.leakage == 'ID' else 9
        search_space = {k: v for k, v in config.items()}  # NO MODIFICATIONS
        
        if config.model_type == 'cnn':
            model = CNN(search_space, poi_width, classes).to(device)
            weight_init(model, search_space.get("kernel_initializer", "glorot_uniform"))
            print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        def leakage_fn(p, k): 
            return AES_Sbox[k ^ int(p)] if config.leakage == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]
        
        # Train the model - SAME AS MAIN
        X_attack_fold = scaler.transform(fold_dataset_obj.X_attack)
        model = attack_driven_training_loop(
            config, model, train_loader, device, run,
            X_attack_fold, fold_dataset_obj.plt_attack, 
            fold_dataset_obj.correct_key, leakage_fn
        )
        
        # Load best model if available - SAME AS MAIN
        model_path = f"best_model_fold_{fold}.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
                print(f"Loaded best model for fold {fold+1}")
            except Exception as e:
                print(f"Could not load model, using last epoch model instead: {e}")
        
        # Save the config as a .npy file - SAME AS MAIN
        np.save("best_model_config.npy", dict(config))

        # Evaluate the model on the attack set - SAME AS MAIN
        X_attack_scaled = scaler.transform(fold_dataset_obj.X_attack)
    
        GE, NTGE, final_ge = evaluate(device, model, X_attack_scaled, fold_dataset_obj.plt_attack, fold_dataset_obj.correct_key,
                                      leakage_fn=leakage_fn, nb_attacks=100,
                                      total_nb_traces_attacks=config.num_traces_attack,
                                      nb_traces_attacks=config.num_traces_attack)
        
        # Save model and metadata if it achieves GE=0 - SAME AS MAIN
        if final_ge == 0:
            run_id = wandb.run.id if wandb.run else "no_wandb"
            model_filename = f"ge0_simple_model_run_{run_id}_fold_{fold}_ntge_{NTGE}.pth"
            config_filename = f"ge0_simple_config_run_{run_id}_fold_{fold}_ntge_{NTGE}.npy"
            metadata_filename = f"ge0_simple_metadata_run_{run_id}_fold_{fold}_ntge_{NTGE}.json"
            
            try:
                # Save the model state dict
                torch.save(model.state_dict(), model_filename)
                print(f"✅ Successfully saved model: {model_filename}")
            except Exception as e:
                print(f"❌ Error saving model: {e}")
            
            try:
                # Save the configuration - convert wandb config to regular dict
                config_dict = dict(config) if hasattr(config, 'items') else {}
                np.save(config_filename, config_dict)
                print(f"✅ Successfully saved config: {config_filename}")
            except Exception as e:
                print(f"❌ Error saving config: {e}")
            
            print(f"🎯 SAVED GE=0 MODEL: {model_filename} (NTGE={NTGE})")
            print(f"📁 Config: {config_filename}")
            print(f"📋 Metadata: {metadata_filename}")
        
        fold_results_ntge.append(NTGE)
        fold_results_ge.append(GE)
        fold_results_final_ge.append(final_ge)

        if run:
            run.log({
                f"fold_{fold+1}_NTGE": NTGE,
                f"fold_{fold+1}_final_GE": final_ge
            })

    # Log the average metrics across all folds - SAME AS MAIN
    avg_ntge = np.mean(fold_results_ntge)
    avg_final_ge = np.mean(fold_results_final_ge)
    avg_ge = np.mean(fold_results_ge, axis=0)
    
    # Summary of GE=0 models found in this run - SAME AS MAIN
    ge0_folds = [i for i, ge in enumerate(fold_results_final_ge) if ge == 0]
    if ge0_folds:
        if config.k_folds == 1:
            print(f"\n🎯 SUMMARY: Model achieved GE=0 with NTGE={fold_results_ntge[0]}")
        else:
            print(f"\n🎯 SUMMARY: Found {len(ge0_folds)} models with GE=0 in this run:")
        run_id = wandb.run.id if wandb.run else "no_wandb"
        for fold_idx in ge0_folds:
            ntge_val = fold_results_ntge[fold_idx]
            model_file = f"ge0_simple_model_run_{run_id}_fold_{fold_idx}_ntge_{ntge_val}.pth"
            if config.k_folds == 1:
                print(f"  - Model saved as: {model_file}")
            else:
                print(f"  - Fold {fold_idx}: NTGE={ntge_val} -> {model_file}")
    else:
        if config.k_folds == 1:
            print(f"\n❌ Model did not achieve GE=0 in this run")
        else:
            print(f"\n❌ No models achieved GE=0 in this run")
    
    wandb.log({
        "avg_final_GE": avg_final_ge,
        "avg_NTGE": avg_ntge,
        "num_ge0_models": len(ge0_folds)
    })

    wandb.finish()

if __name__ == "__main__":
    config_file = "config-ntge_86825.npy"
    run_experiment_with_npy_config(config_file)
