import os, random, numpy as np, torch, wandb
from copy import deepcopy
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.advanced_net import CNN_LSTM_SCA
from src.trainer import training_loop
from src.utils import evaluate, AES_Sbox

def run_experiment():
    run = wandb.init()
    config = wandb.config
    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # On Windows, multiprocessing works best with num_workers=0, but we'll try with it on.
    # The if __name__ == "__main__": guard is the true fix.
    num_workers = 4 if torch.cuda.is_available() else 0

    dataset_obj = Custom_Dataset(root='../', dataset=config.dataset, leakage=config.leakage,
                                 poi_start=config.poi_start, poi_end=config.poi_end,
                                 train_end=500000, test_end=100000,
                                 transform=transforms.Compose([DataAugmentation(max_shift=config.max_shift, noise_level=config.noise_level), ToTensor_trace()]))
    
    dataset_obj.split_attack_set_validation_test()
    
    train_dataset = deepcopy(dataset_obj); train_dataset.choose_phase("train")
    val_dataset = deepcopy(dataset_obj); val_dataset.transform = transforms.Compose([ToTensor_trace()]); val_dataset.choose_phase("validation")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    poi_width, classes = config.poi_end - config.poi_start, 256
    model_config = {
        'cnn_channels': [config.cnn_channels_1, config.cnn_channels_2, config.cnn_channels_3],
        'cnn_kernels': [config.cnn_kernel_1, config.cnn_kernel_2, config.cnn_kernel_3],
        'use_attention': True, 'lstm_hidden_size': config.lstm_hidden_size,
        'lstm_num_layers': config.lstm_num_layers, 'lstm_dropout': config.lstm_dropout,
        'bidirectional': config.bidirectional, 'fc_hidden': config.fc_hidden, 'dropout': config.dropout,
    }

    model = CNN_LSTM_SCA(model_config, poi_width, classes).to(device)
    
    training_loop(config, model, train_loader, val_loader, device)
    model.load_state_dict(torch.load("best_model.pth"))
    
    val_attack_dataset = deepcopy(dataset_obj); val_attack_dataset.choose_phase("validation")
    X_attack_val_raw, plt_attack_val, correct_key = val_attack_dataset.X, val_attack_dataset.Plaintext, val_attack_dataset.correct_key
    X_attack_val = dataset_obj.scaler.transform(X_attack_val_raw.reshape(-1, X_attack_val_raw.shape[-1])).reshape(X_attack_val_raw.shape)
    
    def leakage_fn(p, k): return AES_Sbox[k ^ int(p)]
    
     # Capture all three returned values from the evaluate function
    GE, NTGE, final_GE = evaluate(device, model, X_attack_val, plt_attack_val, correct_key, leakage_fn=leakage_fn, nb_attacks=50, max_traces=10000)

    ### --- NEW --- ###
    # Log both NTGE and our new final_GE metric to wandb
    wandb.log({
        "final_NTGE": NTGE,
        "final_GE_at_max_traces": final_GE
    })
    ### --- END NEW --- ###

    if GE is not None and len(GE) > 0:
        ge_data = [[x+1, y] for (x, y) in enumerate(GE)]
        table = wandb.Table(data=ge_data, columns=["Traces", "GE"])
        wandb.log({"GE_Curve_Validation": wandb.plot.line(table, "Traces", "GE", title="GE vs. Traces")})
    wandb.finish()

# THIS IS THE CRITICAL FIX FOR MULTIPROCESSING ON WINDOWS
if __name__ == "__main__":
    run_experiment()