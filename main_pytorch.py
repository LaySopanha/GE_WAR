# main_pytorch.py
import os, random, numpy as np, torch, wandb
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
from src.trainer import training_loop
from src.utils import evaluate, AES_Sbox, calculate_snr

def run_experiment():
    run = wandb.init()
    config = wandb.config
    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load full traces first to compute SNR
    full_dataset_obj = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage,
                                      poi_start=0, poi_end=7000,  # Load all points
                                      train_end=500000, test_end=100000) # Use a subset for SNR calc

    # Calculate SNR on the full profiling dataset for accuracy
    snr = calculate_snr(full_dataset_obj.X_profiling, full_dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr)[-config.num_poi:]

    # Create a new dataset object with only the selected POIs
    dataset_obj = deepcopy(full_dataset_obj)
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]

    # Randomly select a subset of the profiling data for hyperparameter tuning
    profiling_indices = np.random.choice(len(dataset_obj.X_profiling), 200000, replace=False)
    dataset_obj.X_profiling = dataset_obj.X_profiling[profiling_indices]
    dataset_obj.Y_profiling = dataset_obj.Y_profiling[profiling_indices]

    # K-Fold Cross-Validation
    kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=SEED)
    fold_results_ntge = []
    fold_results_ge = []
    fold_results_final_ge = []

    for fold, (train_index, val_index) in enumerate(kf.split(dataset_obj.X_profiling)):
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
        model = CNN(search_space, poi_width, classes).to(device)
        weight_init(model, search_space.get("kernel_initializer", "glorot_uniform"))

        # Train the model
        model = training_loop(config, model, train_loader, val_loader, device, run)
        model.load_state_dict(torch.load("best_model.pth"))
        
        # Save the config as a .npy file
        np.save("best_model_config.npy", config)

        # Evaluate the model on the attack set
        attack_scaler = StandardScaler()
        X_attack_scaled = attack_scaler.fit_transform(fold_dataset_obj.X_attack)
        
        def leakage_fn(p, k): return AES_Sbox[k ^ int(p)] if config.leakage == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]
        
        GE, NTGE, final_ge = evaluate(device, model, X_attack_scaled, fold_dataset_obj.plt_attack, fold_dataset_obj.correct_key,
                                      leakage_fn=leakage_fn, nb_attacks=100,
                                      total_nb_traces_attacks=config.num_traces_attack,
                                      nb_traces_attacks=config.num_traces_attack)
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
    
    # Combined metric for sweep optimization
    combined_metric = avg_final_ge + avg_ntge

    wandb.log({
        "final_NTGE": avg_ntge,
        "final_GE_at_max_traces": avg_final_ge,
        "combined_metric": combined_metric
    })

    if avg_ge is not None and len(avg_ge) > 0:
        ge_data = [[i, ge] for i, ge in enumerate(avg_ge)]
        table = wandb.Table(data=ge_data, columns=["Trace", "GE"])
        wandb.log({"GE_Curve_Data": table})

    wandb.finish()

if __name__ == "__main__":
    run_experiment()
