#!/usr/bin/env python3
"""
Train model using NPY config file
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
    """Run experiment using NPY config file"""
    
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

    # Calculate SNR on the full profiling dataset for accuracy
    snr = calculate_snr(full_dataset_obj.X_profiling, full_dataset_obj.Y_profiling)
    
    # GENERALIZATION IMPROVEMENT 1: Slightly more POIs for better coverage
    # But not too aggressive - your 100 POI approach already works!
    num_poi_enhanced = min(150, int(config.num_poi * 1.2))  # Only 20% more POIs
    top_k_indices = np.argsort(snr)[-num_poi_enhanced:]
    
    print(f"📍 Enhanced POI selection: Using {num_poi_enhanced} POIs (vs {config.num_poi} original)")
    print(f"📊 SNR range: {snr[top_k_indices].min():.3f} - {snr[top_k_indices].max():.3f}")

    # Create a new dataset object with only the selected POIs
    dataset_obj = deepcopy(full_dataset_obj)
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]
    
    # GENERALIZATION IMPROVEMENT 2: Save POI indices for consistent evaluation
    poi_metadata = {
        'poi_indices': top_k_indices.tolist(),
        'snr_values': snr[top_k_indices].tolist(),
        'num_poi_used': num_poi_enhanced,
        'original_poi_config': config.num_poi
    }
    np.save('poi_selection_metadata.npy', poi_metadata)

    # Use the full profiling dataset (500k) instead of random subset
    print(f"📈 Using full {len(dataset_obj.X_profiling)} profiling traces (vs 200k subset)")

    # Handle single split vs k-fold cross-validation
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

        # Create a scaler for this fold
        scaler = StandardScaler()
        
        # Create datasets for this fold
        train_transform = transforms.Compose([
            DataAugmentation(max_shift=config.max_shift, noise_level=config.noise_level),
            ToTensor_trace()])
        eval_transform = transforms.Compose([ToTensor_trace()])

        # Create a deepcopy of the dataset object for this fold
        fold_dataset_obj = deepcopy(dataset_obj)
        
        # Get the training and validation data for this fold
        X_train, X_val = fold_dataset_obj.X_profiling[train_index], fold_dataset_obj.X_profiling[val_index]
        Y_train, Y_val = fold_dataset_obj.Y_profiling[train_index], fold_dataset_obj.Y_profiling[val_index]

        # Fit the scaler on the training data for this fold
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Create the dataloaders for this fold
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

        # Initialize model for this fold
        poi_width = num_poi_enhanced  # Use enhanced POI count
        classes = 256 if config.leakage == 'ID' else 9
        search_space = {k: v for k, v in config.items()}
        
        # GENERALIZATION IMPROVEMENT 3: Conservative regularization enhancement
        # Keep original dropout but add small weight decay
        original_dropout = search_space.get('dropout_rate', 0.1)
        search_space['dropout_rate'] = max(original_dropout, 0.17)  # Only slightly higher
        search_space['weight_decay'] = search_space.get('weight_decay', 1e-5)  # Small L2 reg
        
        if config.model_type == 'cnn':
            model = CNN(search_space, poi_width, classes).to(device)
            weight_init(model, search_space.get("kernel_initializer", "glorot_uniform"))
            print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
            print(f"🛡️ Enhanced regularization: dropout={search_space['dropout_rate']:.3f}")
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        def leakage_fn(p, k): 
            return AES_Sbox[k ^ int(p)] if config.leakage == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]
        
        # Train the model
        X_attack_fold = scaler.transform(fold_dataset_obj.X_attack)
        model = attack_driven_training_loop(
            config, model, train_loader, device, run,
            X_attack_fold, fold_dataset_obj.plt_attack, 
            fold_dataset_obj.correct_key, leakage_fn
        )
        
        # Load best model if available
        model_path = f"best_model_fold_{fold}.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
                print(f"Loaded best model for fold {fold+1}")
            except Exception as e:
                print(f"Could not load model, using last epoch model instead: {e}")
        
        # Save the config as a .npy file
        np.save("best_model_config.npy", dict(config))

        # GENERALIZATION IMPROVEMENT 4: Multiple evaluation strategies
        X_attack_scaled = scaler.transform(fold_dataset_obj.X_attack)
        
        # Standard evaluation
        GE, NTGE, final_ge = evaluate(device, model, X_attack_scaled, fold_dataset_obj.plt_attack, fold_dataset_obj.correct_key,
                                      leakage_fn=leakage_fn, nb_attacks=100,
                                      total_nb_traces_attacks=config.num_traces_attack,
                                      nb_traces_attacks=config.num_traces_attack)
        
        # GENERALIZATION IMPROVEMENT 5: Noise robustness test
        # Add small amount of gaussian noise to test robustness
        noise_levels = [0.01, 0.02, 0.05]
        robust_scores = []
        
        for noise_level in noise_levels:
            X_attack_noisy = X_attack_scaled + np.random.normal(0, noise_level, X_attack_scaled.shape)
            _, ntge_noisy, ge_noisy = evaluate(device, model, X_attack_noisy, fold_dataset_obj.plt_attack, fold_dataset_obj.correct_key,
                                             leakage_fn=leakage_fn, nb_attacks=50,  # Fewer attacks for speed
                                             total_nb_traces_attacks=config.num_traces_attack,
                                             nb_traces_attacks=config.num_traces_attack)
            robust_scores.append(ge_noisy)
            print(f"🔊 Noise robustness (σ={noise_level}): GE={ge_noisy}")
        
        # Calculate robustness score (lower is better)
        avg_robust_ge = np.mean(robust_scores)
        robustness_penalty = avg_robust_ge - final_ge
        print(f"🎯 Robustness penalty: +{robustness_penalty:.1f} GE under noise")
        
        # GENERALIZATION IMPROVEMENT 6: Save ALL GE=0 models (like main_pytorch.py)
        # But also test and log robustness for analysis
        generalization_score = final_ge + 0.1 * robustness_penalty  # For logging only
        
        if final_ge == 0:  # Save ANY GE=0 model (like original main)
            run_id = wandb.run.id if wandb.run else "no_wandb"
            
            # Create both robust and standard filenames for compatibility
            if robustness_penalty < 5:  # Mark truly robust ones
                model_filename = f"ge0_robust_model_run_{run_id}_fold_{fold}_ntge_{NTGE}_robust_{robustness_penalty:.1f}.pth"
                print(f"🎯 EXCELLENT: GE=0 with good robustness (penalty={robustness_penalty:.1f})")
            else:
                model_filename = f"ge0_model_run_{run_id}_fold_{fold}_ntge_{NTGE}_robust_{robustness_penalty:.1f}.pth"
                print(f"⚠️ ACCEPTABLE: GE=0 but higher noise sensitivity (penalty={robustness_penalty:.1f})")
            
            config_filename = f"ge0_config_run_{run_id}_fold_{fold}_ntge_{NTGE}.npy"
            metadata_filename = f"ge0_metadata_run_{run_id}_fold_{fold}_ntge_{NTGE}.json"
            
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
            
            # Save comprehensive metadata
            import json
            
            # Helper function to convert numpy types to native Python types
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_json_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                else:
                    return obj
            
            metadata = {
                'run_id': str(run_id),
                'fold': int(fold),
                'final_GE': convert_to_json_serializable(final_ge),
                'NTGE': convert_to_json_serializable(NTGE),
                'num_poi': convert_to_json_serializable(num_poi_enhanced),
                'original_poi_config': convert_to_json_serializable(config.num_poi),
                'train_traces': convert_to_json_serializable(train_end),
                'leakage': str(config.leakage),
                'dataset': str(config.dataset),
                'model_type': str(config.model_type),
                'batch_size': convert_to_json_serializable(config.batch_size),
                'learning_rate': convert_to_json_serializable(getattr(config, 'lr', getattr(config, 'learning_rate', 0.0))),
                'poi_indices': convert_to_json_serializable(top_k_indices),
                'model_parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                'robustness_penalty': convert_to_json_serializable(robustness_penalty),
                'noise_robustness_scores': convert_to_json_serializable(robust_scores),
                'generalization_score': convert_to_json_serializable(generalization_score)
            }
            
            try:
                with open(metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✅ Successfully saved metadata: {metadata_filename}")
            except Exception as e:
                print(f"❌ Error saving metadata: {e}")
                # Try to save without problematic fields
                safe_metadata = {
                    'run_id': str(run_id),
                    'fold': int(fold),
                    'final_GE': 0.0,
                    'NTGE': int(NTGE) if isinstance(NTGE, (int, np.integer)) else int(float(NTGE)),
                    'num_poi': int(num_poi_enhanced),
                    'original_poi_config': int(config.num_poi),
                    'train_traces': int(train_end),
                    'leakage': str(config.leakage),
                    'dataset': str(config.dataset),
                    'robustness_penalty': float(robustness_penalty)
                }
                with open(metadata_filename.replace('.json', '_safe.json'), 'w') as f:
                    json.dump(safe_metadata, f, indent=2)
                print(f"✅ Saved safe metadata: {metadata_filename.replace('.json', '_safe.json')}")
            
            print(f"🎯 SAVED GE=0 MODEL: {model_filename} (NTGE={NTGE}, Robustness={robustness_penalty:.1f})")
            print(f"📁 Config: {config_filename}")
            print(f"📋 Metadata: {metadata_filename}")
        else:
            print(f"❌ Model did not achieve GE=0 (final_ge={final_ge})")
        
        fold_results_ntge.append(NTGE)
        fold_results_ge.append(GE)
        fold_results_final_ge.append(final_ge)
        
        # Log additional metrics to wandb
        if run:
            run.log({
                f"fold_{fold+1}_NTGE": NTGE,
                f"fold_{fold+1}_final_GE": final_ge,
                f"fold_{fold+1}_robustness_penalty": robustness_penalty,
                f"fold_{fold+1}_generalization_score": generalization_score
            })

    # Log the average metrics across all folds
    avg_ntge = np.mean(fold_results_ntge)
    avg_final_ge = np.mean(fold_results_final_ge)
    avg_ge = np.mean(fold_results_ge, axis=0)
    
    # Improved selection: find lowest NTGE among all GE=0 folds
    ge0_indices = [i for i, ge in enumerate(fold_results_final_ge) if ge == 0]
    if ge0_indices:
        # If any fold achieved GE=0, find the one with lowest NTGE
        best_final_ge = 0
        best_ntge = min([fold_results_ntge[i] for i in ge0_indices])
        best_fold_index = ge0_indices[np.argmin([fold_results_ntge[i] for i in ge0_indices])]
        
        print(f"DEBUG: Found {len(ge0_indices)} folds with GE=0")
        print(f"DEBUG: Fold indices with GE=0: {ge0_indices}")
        print(f"DEBUG: NTGE values for these folds: {[fold_results_ntge[i] for i in ge0_indices]}")
        print(f"DEBUG: Selected fold {best_fold_index} with NTGE={best_ntge}")
    else:
        # If no fold achieved GE=0, take the fold with lowest GE
        best_final_ge = min(fold_results_final_ge)
        best_fold_index = fold_results_final_ge.index(best_final_ge)
        best_ntge = fold_results_ntge[best_fold_index]
        
        print(f"DEBUG: No folds achieved GE=0")
        print(f"DEBUG: All fold GE values: {fold_results_final_ge}")
        print(f"DEBUG: Selected fold {best_fold_index} with GE={best_final_ge}")

    # Add a print for the composite score calculation
    composite_score = 1000000 * best_final_ge + (best_ntge if best_final_ge == 0 else 0)
    print(f"DEBUG: Composite score calculation: 1000000 * {best_final_ge} + {best_ntge if best_final_ge == 0 else 0} = {composite_score}")
    
    # Summary of GE=0 models found in this run
    ge0_folds = [i for i, ge in enumerate(fold_results_final_ge) if ge == 0]
    if ge0_folds:
        if config.k_folds == 1:
            print(f"\n🎯 SUMMARY: Model achieved GE=0 with NTGE={fold_results_ntge[0]}")
        else:
            print(f"\n🎯 SUMMARY: Found {len(ge0_folds)} models with GE=0 in this run:")
        run_id = wandb.run.id if wandb.run else "no_wandb"
        for fold_idx in ge0_folds:
            ntge_val = fold_results_ntge[fold_idx]
            model_file = f"ge0_model_run_{run_id}_fold_{fold_idx}_ntge_{ntge_val}.pth"
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
        "best_final_GE": best_final_ge,
        "best_NTGE": best_ntge,
        "avg_final_GE": avg_final_ge,
        "avg_NTGE": avg_ntge,
        "composite_score": composite_score,
        "num_ge0_models": len(ge0_folds)
    })

    if avg_ge is not None and len(avg_ge) > 0:
        ge_data = [[i, ge] for i, ge in enumerate(avg_ge)]
        table = wandb.Table(data=ge_data, columns=["Trace", "GE"])
        wandb.log({"GE_Curve_Data": table})

    wandb.finish()

if __name__ == "__main__":
    config_file = "config-ntge_86825.npy"
    run_experiment_with_npy_config(config_file)
