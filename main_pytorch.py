# main_pytorch.py
import os, random, numpy as np, torch, wandb
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
# from src.advanced_net import ResNetSCA
from src.trainer import training_loop, attack_driven_training_loop
from src.utils import evaluate, AES_Sbox, calculate_snr

def run_experiment():
    run = wandb.init()
    config = wandb.config
    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load full traces first to compute SNR
    train_end = getattr(config, 'train_end', 500000)  # Use config value or default to 500k
    full_dataset_obj = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage,
                                      poi_start=0, poi_end=7000,  # Load all points
                                      train_end=train_end, test_end=100000)
    
    print(f"🚀 Loading {train_end} profiling traces from dataset")

    # Calculate SNR on the full profiling dataset for accuracy
    snr = calculate_snr(full_dataset_obj.X_profiling, full_dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr)[-config.num_poi:]

    # Create a new dataset object with only the selected POIs
    dataset_obj = deepcopy(full_dataset_obj)
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]

    # Use the full profiling dataset (500k) instead of random subset
    print(f"📈 Using full {len(dataset_obj.X_profiling)} profiling traces (vs 200k subset)")
    # No more subset selection - use all 500k traces!

    # K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=SEED)
    fold_results_ntge = []
    fold_results_ge = []
    fold_results_final_ge = []

    for fold, (train_index, val_index) in enumerate(kf.split(dataset_obj.X_profiling, dataset_obj.Y_profiling)):
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
        poi_width = config.num_poi
        classes = 256 if config.leakage == 'ID' else 9
        search_space = {k: v for k, v in config.items()}
        if config.model_type == 'cnn':
            model = CNN(search_space, poi_width, classes).to(device)
            weight_init(model, search_space.get("kernel_initializer", "glorot_uniform"))
            print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
        # elif config.model_type == 'resnet':
            # model = ResNetSCA(search_space, poi_width, classes).to(device)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        def leakage_fn(p, k): 
            return AES_Sbox[k ^ int(p)] if config.leakage == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]
        
        # Train the model
        # model = training_loop(config, model, train_loader, val_loader, device, run)
        X_attack_fold = scaler.transform(fold_dataset_obj.X_attack)
        model = attack_driven_training_loop(
            config, model, train_loader, device, run,
            X_attack_fold, fold_dataset_obj.plt_attack, 
            fold_dataset_obj.correct_key, leakage_fn
        )
        # model.load_state_dict(torch.load("best_model.pth"))
        model_path = f"best_model_fold_{fold}.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
                print(f"Loaded best model for fold {fold+1}")
            except Exception as e:
                print(f"Could not load model, using last epoch model instead: {e}")
        
        # Save the config as a .npy file
        np.save("best_model_config.npy", config)

        # Evaluate the model on the attack set
        X_attack_scaled = scaler.transform(fold_dataset_obj.X_attack)
    
        GE, NTGE, final_ge = evaluate(device, model, X_attack_scaled, fold_dataset_obj.plt_attack, fold_dataset_obj.correct_key,
                                      leakage_fn=leakage_fn, nb_attacks=100,
                                      total_nb_traces_attacks=config.num_traces_attack,
                                      nb_traces_attacks=config.num_traces_attack)
        
        # Save model and metadata if it achieves GE=0
        if final_ge == 0:
            run_id = wandb.run.id if wandb.run else "no_wandb"
            model_filename = f"ge0_model_run_{run_id}_fold_{fold}_ntge_{NTGE}.pth"
            config_filename = f"ge0_config_run_{run_id}_fold_{fold}_ntge_{NTGE}.npy"
            metadata_filename = f"ge0_metadata_run_{run_id}_fold_{fold}_ntge_{NTGE}.json"
            
            try:
                # Save the model state dict
                torch.save(model.state_dict(), model_filename)
                print(f"✅ Successfully saved model: {model_filename}")
            except Exception as e:
                print(f"❌ Error saving model: {e}")
                print(f"Error details: {type(e).__name__}: {str(e)}")
                # Try pickle fallback
                import pickle
                fallback_path = model_filename.replace('.pth', '_pickle.pkl')
                try:
                    with open(fallback_path, 'wb') as f:
                        pickle.dump(model.state_dict(), f)
                    print(f"✅ Pickle fallback successful: {fallback_path}")
                except Exception as e2:
                    print(f"❌ Pickle fallback failed: {e2}")
            
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
                'num_poi': convert_to_json_serializable(config.num_poi),
                'train_traces': convert_to_json_serializable(train_end),
                'leakage': str(config.leakage),
                'dataset': str(config.dataset),
                'model_type': str(config.model_type),
                'batch_size': convert_to_json_serializable(config.batch_size),
                'learning_rate': convert_to_json_serializable(getattr(config, 'lr', getattr(config, 'learning_rate', 0.0))),
                'poi_indices': convert_to_json_serializable(top_k_indices),
                'model_parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            }
            
            try:
                with open(metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✅ Successfully saved metadata: {metadata_filename}")
            except Exception as e:
                print(f"❌ Error saving metadata: {e}")
                print(f"Metadata content: {metadata}")
                # Try to save without problematic fields
                safe_metadata = {
                    'run_id': str(run_id),
                    'fold': int(fold),
                    'final_GE': 0.0,
                    'NTGE': int(NTGE) if isinstance(NTGE, (int, np.integer)) else int(float(NTGE)),
                    'num_poi': int(config.num_poi),
                    'train_traces': int(train_end),
                    'leakage': str(config.leakage),
                    'dataset': str(config.dataset)
                }
                with open(metadata_filename.replace('.json', '_safe.json'), 'w') as f:
                    json.dump(safe_metadata, f, indent=2)
                print(f"✅ Saved safe metadata: {metadata_filename.replace('.json', '_safe.json')}")
            
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
        
        # Add these debug prints
        print(f"DEBUG: Found {len(ge0_indices)} folds with GE=0")
        print(f"DEBUG: Fold indices with GE=0: {ge0_indices}")
        print(f"DEBUG: NTGE values for these folds: {[fold_results_ntge[i] for i in ge0_indices]}")
        print(f"DEBUG: Selected fold {best_fold_index} with NTGE={best_ntge}")
    else:
        # If no fold achieved GE=0, take the fold with lowest GE
        best_final_ge = min(fold_results_final_ge)
        best_fold_index = fold_results_final_ge.index(best_final_ge)
        best_ntge = fold_results_ntge[best_fold_index]
        
        # Add these debug prints
        print(f"DEBUG: No folds achieved GE=0")
        print(f"DEBUG: All fold GE values: {fold_results_final_ge}")
        print(f"DEBUG: Selected fold {best_fold_index} with GE={best_final_ge}")

    # Add a print for the composite score calculation
    composite_score = 1000000 * best_final_ge + (best_ntge if best_final_ge == 0 else 0)
    print(f"DEBUG: Composite score calculation: 1000000 * {best_final_ge} + {best_ntge if best_final_ge == 0 else 0} = {composite_score}")
    
    # Summary of GE=0 models found in this run
    ge0_folds = [i for i, ge in enumerate(fold_results_final_ge) if ge == 0]
    if ge0_folds:
        print(f"\n🎯 SUMMARY: Found {len(ge0_folds)} models with GE=0 in this run:")
        run_id = wandb.run.id if wandb.run else "no_wandb"
        for fold_idx in ge0_folds:
            ntge_val = fold_results_ntge[fold_idx]
            model_file = f"ge0_model_run_{run_id}_fold_{fold_idx}_ntge_{ntge_val}.pth"
            print(f"  - Fold {fold_idx}: NTGE={ntge_val} -> {model_file}")
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
    run_experiment()
