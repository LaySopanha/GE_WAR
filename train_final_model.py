# train_final_model.py
import os, random, numpy as np, torch, yaml
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
from src.trainer import training_loop
from src.utils import evaluate, AES_Sbox, calculate_snr

def train_final_model():
    # Load the best hyperparameters from the sweep
    with open("best_model_config.npy", "r") as f:
        config = yaml.safe_load(f)

    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the full dataset
    dataset_obj = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage,
                                 poi_start=0, poi_end=7000,
                                 train_end=500000, test_end=100000)

    # Calculate SNR on the full profiling dataset for accuracy
    snr = calculate_snr(dataset_obj.X_profiling, dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr)[-config.num_poi:]

    # Create a new dataset object with only the selected POIs
    final_dataset_obj = deepcopy(dataset_obj)
    final_dataset_obj.X_profiling = final_dataset_obj.X_profiling[:, top_k_indices]
    final_dataset_obj.X_attack = final_dataset_obj.X_attack[:, top_k_indices]

    # Create a scaler
    scaler = StandardScaler()

    # Create datasets
    train_transform = transforms.Compose([
        DataAugmentation(max_shift=config.max_shift, noise_level=config.noise_level),
        ToTensor_trace()])
    eval_transform = transforms.Compose([ToTensor_trace()])

    # Get the training data
    X_train, Y_train = final_dataset_obj.X_profiling, final_dataset_obj.Y_profiling

    # Fit the scaler on the training data
    X_train = scaler.fit_transform(X_train)

    # Create the dataloaders
    train_dataset = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage)
    train_dataset.X_profiling, train_dataset.Y_profiling = X_train, Y_train
    train_dataset.transform = train_transform
    train_dataset.choose_phase("train")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # Initialize model
    poi_width = config.num_poi
    classes = 256 if config.leakage == 'ID' else 9
    search_space = {k: v for k, v in config.items()}
    model = CNN(search_space, poi_width, classes).to(device)
    weight_init(model, search_space.get("kernel_initializer", "glorot_uniform"))

    # Train the model
    run = wandb.init(project=PROJECT, config=config)
    model = training_loop(config, model, train_loader, None, device, run) # No validation loader needed for final training
    torch.save(model.state_dict(), "final_model.pth")

    # Evaluate the model on the attack set
    attack_scaler = StandardScaler()
    X_attack_scaled = attack_scaler.fit_transform(final_dataset_obj.X_attack)

    def leakage_fn(p, k): return AES_Sbox[k ^ int(p)] if config.leakage == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]

    GE, NTGE, final_ge = evaluate(device, model, X_attack_scaled, final_dataset_obj.plt_attack, final_dataset_obj.correct_key,
                                  nb_attacks=100,
                                  nb_traces_attacks=config.num_traces_attack,
                                  leakage_fn=leakage_fn)

    print(f"Final NTGE: {NTGE}")
    print(f"Final GE at max traces: {final_ge}")

if __name__ == "__main__":
    train_final_model()
