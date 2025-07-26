import os, random, numpy as np, torch, wandb
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
from src.trainer import attack_driven_training_loop
from src.utils import evaluate, AES_Sbox, calculate_snr

def train_final_model():
    # --- Best Hyperparameters (TO BE FILLED IN LATER) ---
    config = {
        "activation": "selu",
        "batch_size": 256,
        "conv_layers": 2,
        "dataset": "CHES_2025",
        "dropout_rate": 0.25,
        "epochs": 200,
        "filters": 8,
        "kernel_initializer": "glorot_uniform",
        "kernels": 36,
        "layers": 2,
        "leakage": "ID",
        "lr": 0.005,
        "max_shift": 20,
        "model_type": "cnn",
        "neurons": 256,
        "noise_level": 0.07659424572657016,
        "num_poi": 50,
        "num_traces_attack": 100000,
        "optimizer": "Adam",
        "padding": 0,
        "poi_end": 4519,
        "poi_start": 4119,
        "pooling_sizes": 3,
        "pooling_types": "average_pool",
    }

    run = wandb.init(project="GE_WAR_Final_Models", config=config, job_type="final-training")
    
    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Data Loading and Preparation ---
    full_dataset_obj = Custom_Dataset(root='../', dataset=config["dataset"], leakage=config["leakage"],
                                      poi_start=0, poi_end=7000,
                                      train_end=500000, test_end=100000)

    snr = calculate_snr(full_dataset_obj.X_profiling, full_dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr)[-config["num_poi"]:]

    dataset_obj = deepcopy(full_dataset_obj)
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]

    scaler = StandardScaler()
    dataset_obj.X_profiling = scaler.fit_transform(dataset_obj.X_profiling)
    dataset_obj.X_attack = scaler.transform(dataset_obj.X_attack)

    train_transform = transforms.Compose([
        DataAugmentation(max_shift=config["max_shift"], noise_level=config["noise_level"]),
        ToTensor_trace()])

    train_dataset = Custom_Dataset(root='../', dataset=config["dataset"], leakage=config["leakage"])
    train_dataset.X_profiling, train_dataset.Y_profiling = dataset_obj.X_profiling, dataset_obj.Y_profiling
    train_dataset.transform = train_transform
    train_dataset.choose_phase("train")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    # --- Model Initialization and Training ---
    poi_width = config["num_poi"]
    classes = 256 if config["leakage"] == 'ID' else 9
    model = CNN(config, poi_width, classes).to(device)
    weight_init(model, config["kernel_initializer"])

    def leakage_fn(p, k): return AES_Sbox[k ^ int(p)] if config["leakage"] == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]

    model = attack_driven_training_loop(config, model, train_loader, device, run, 
                                        dataset_obj.X_attack, dataset_obj.plt_attack, 
                                        dataset_obj.correct_key, leakage_fn)
    
    # Load the best model saved by the training loop
    model.load_state_dict(torch.load("best_model.pth"))
    
    # Save the fully trained model
    torch.save(model.state_dict(), "fully_trained_model.pth")
    wandb.save("fully_trained_model.pth")

    wandb.finish()

if __name__ == "__main__":
    train_final_model()
