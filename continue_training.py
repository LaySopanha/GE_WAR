# main_pytorch.py
import os, random, numpy as np, torch
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
from src.trainer import training_loop
from src.utils import evaluate, AES_Sbox, calculate_snr

def run_experiment():
    # Hardcoded config for continuing training
    config = {
        "dataset": "CHES_2025",
        "leakage": "ID",
        "num_poi": 100,
        "max_shift": 10,
        "noise_level": 0.03505419099952169,
        "batch_size": 128,
        "lr": 0.001,
        "epochs": 100, # Increased for full training
        "optimizer": "Adam",
        "conv_layers": 3,
        "filters": 8,
        "kernels": 12,
        "activation": "relu",
        "pooling_types": "average_pool",
        "pooling_sizes": 4,
        "padding": 0,
        "layers": 2,
        "neurons": 256,
        "kernel_initializer": "glorot_uniform"
    }
    
    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load full traces
    full_dataset_obj = Custom_Dataset(root='../', dataset=config['dataset'], leakage=config['leakage'],
                                      poi_start=0, poi_end=7000,
                                      train_end=500000, test_end=100000)
    
    # Calculate SNR and select top POIs
    snr = calculate_snr(full_dataset_obj.X_profiling, full_dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr)[-config['num_poi']:]
    
    dataset_obj = full_dataset_obj
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]
    
    scaler = StandardScaler()
    dataset_obj.X_profiling = scaler.fit_transform(dataset_obj.X_profiling)
    dataset_obj.X_attack = scaler.transform(dataset_obj.X_attack)
    dataset_obj.scaler = scaler
    
    print(f"Selected {config['num_poi']} POIs based on SNR.")
    
    dataset_obj.split_attack_set_validation_test()
    
    train_transform = transforms.Compose([
        DataAugmentation(max_shift=config['max_shift'], noise_level=config['noise_level']),
        ToTensor_trace()])
    eval_transform = transforms.Compose([ToTensor_trace()])
    
    train_dataset = deepcopy(dataset_obj); train_dataset.transform = train_transform; train_dataset.choose_phase("train")
    val_dataset = deepcopy(dataset_obj); val_dataset.transform = eval_transform; val_dataset.choose_phase("validation")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    poi_width = config['num_poi']
    classes = 256 if config['leakage'] == 'ID' else 9
    model = CNN(config, poi_width, classes).to(device)
    weight_init(model, config.get("kernel_initializer", "glorot_uniform"))
    
    # Load the best model from the sweep
    model.load_state_dict(torch.load("Result/CHES_2025_cnn_ID/models/best_model_2.pth"))
    
    model = training_loop(config, model, train_loader, val_loader, device)
    
    # Save the fully trained model
    torch.save(model.state_dict(), "Result/CHES_2025_cnn_ID/models/fully_trained_model_2.pth")
    
    # val_attack_dataset = deepcopy(dataset_obj); val_attack_dataset.transform = eval_transform; val_attack_dataset.choose_phase("validation")
    # X_attack_val_raw, plt_attack_val, correct_key = val_attack_dataset.X, val_attack_dataset.Plaintext, val_attack_dataset.correct_key
    # X_attack_val = dataset_obj.scaler.transform(X_attack_val_raw)
    
    # def leakage_fn(p, k): return AES_Sbox[k ^ int(p)] if config['leakage'] == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]
    
    # GE, NTGE, final_ge = evaluate(device, model, X_attack_val, plt_attack_val, correct_key, 
    #                               leakage_fn=leakage_fn, nb_attacks=100,
    #                               total_nb_traces_attacks=100000, nb_traces_attacks=100000)

    # print(f"Final NTGE: {NTGE}")
    # print(f"Final GE at max traces: {final_ge}")
    # print(f"Full GE: {GE}")

if __name__ == "__main__":
    run_experiment()
