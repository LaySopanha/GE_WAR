# main_pytorch.py
import os, random, numpy as np, torch, wandb
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
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
                                      train_end=50000, test_end=100000) # Use a subset for SNR calc
    
    # Calculate SNR and select top POIs
    snr = calculate_snr(full_dataset_obj.X_profiling, full_dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr)[-config.num_poi:]
    
    # Create a new dataset object with only the selected POIs
    # This is a simplified way; a more robust implementation might modify Custom_Dataset
    # For now, we manually slice the data arrays for speed of implementation.
    dataset_obj = full_dataset_obj
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]
    
    # Fit the scaler on the training data *after* POI selection
    scaler = StandardScaler()
    dataset_obj.X_profiling = scaler.fit_transform(dataset_obj.X_profiling)
    dataset_obj.X_attack = scaler.transform(dataset_obj.X_attack)
    dataset_obj.scaler = scaler  # Store the scaler in the dataset object
    
    print(f"Selected {config.num_poi} POIs based on SNR.")
    
    dataset_obj.split_attack_set_validation_test()
    
    train_transform = transforms.Compose([
        DataAugmentation(max_shift=config.max_shift, noise_level=config.noise_level),
        ToTensor_trace()])
    eval_transform = transforms.Compose([ToTensor_trace()])
    
    train_dataset = deepcopy(dataset_obj); train_dataset.transform = train_transform; train_dataset.choose_phase("train")
    val_dataset = deepcopy(dataset_obj); val_dataset.transform = eval_transform; val_dataset.choose_phase("validation")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    poi_width = config.num_poi # Use the number of selected POIs
    classes = 256 if config.leakage == 'ID' else 9
    search_space = {k: v for k, v in config.items()}
    model = CNN(search_space, poi_width, classes).to(device)
    weight_init(model, search_space.get("kernel_initializer", "glorot_uniform"))

    model = training_loop(config, model, train_loader, val_loader, device)
    
    model.load_state_dict(torch.load("best_model.pth"))
    
    val_attack_dataset = deepcopy(dataset_obj); val_attack_dataset.transform = eval_transform; val_attack_dataset.choose_phase("validation")
    X_attack_val_raw, plt_attack_val, correct_key = val_attack_dataset.X, val_attack_dataset.Plaintext, val_attack_dataset.correct_key
    X_attack_val = dataset_obj.scaler.transform(X_attack_val_raw)
    
    def leakage_fn(p, k): return AES_Sbox[k ^ int(p)] if config.leakage == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]
    
    # For the rapid sweep, use fewer traces to speed up evaluation
    GE, NTGE, final_ge = evaluate(device, model, X_attack_val, plt_attack_val, correct_key, 
                                  leakage_fn=leakage_fn, nb_attacks=50,
                                  total_nb_traces_attacks=10000, nb_traces_attacks=10000)

    wandb.log({
        "final_NTGE": NTGE,
        "final_GE_at_max_traces": final_ge
    })

    if GE is not None and len(GE) > 0:
        ge_data = [[i, round(ge, 2)] for i, ge in enumerate(GE)]
        table = wandb.Table(data=ge_data, columns=["Traces", "GE"])
        wandb.log({"GE vs. Traces": wandb.plot.line(table, "Traces", "GE", title="GE vs. Traces")})
    wandb.finish()

if __name__ == "__main__":
    run_experiment()
