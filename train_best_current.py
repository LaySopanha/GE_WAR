# train_best_current.py - Train the best model from current sweep for submission
import os, random, numpy as np, torch, wandb
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
from src.trainer import training_loop, attack_driven_training_loop
from src.utils import evaluate, AES_Sbox, calculate_snr

def train_best_model():
    """Train the best configuration from sweep 48lz9a53 with extended training"""
    
    # === CONFIGURATION FROM BEST SWEEP RESULT ===
    # ACTUAL BEST HYPERPARAMETERS from run_1jcrispy (NTGE=97397)
    config = {
        # Model Architecture - Keep same as sweep
        'model_type': 'cnn',
        'dropout_rate': 0.2548776576936158,  # ACTUAL BEST VALUE
        'layers': 2,
        'neurons': 512,
        'activation': 'relu',
        'conv_layers': 4,
        'filters': 32,
        'kernels': 24,
        'pooling_types': 'max_pool',
        'pooling_sizes': 2,
        'padding': 0,
        
        # Training - EXTENDED FOR BETTER NTGE
        'early_stopping_patience': 30,  # More patience for final training
        'min_epochs': 80,  # Ensure adequate training
        'attack_eval_frequency': 15,  # More frequent evaluation
        'epochs': 400,  # Extended training
        
        # Optimization - Use best from sweep
        'lr': 0.0001,  # ACTUAL BEST VALUE
        'optimizer': 'Adam',
        'batch_size': 32,
        
        # Augmentation - Use best from sweep
        'max_shift': 20,
        'noise_level': 0.01969817242376655,  # ACTUAL BEST VALUE
        
        # Dataset
        'dataset': 'CHES_2025',
        'leakage': 'ID',
        'num_poi': 250,
        'poi_start': 0,
        'poi_end': 7000,
        'train_end': 350000,
        'num_traces_attack': 100000,
        'k_folds': 1
    }
    
    print(f"🎯 Training FINAL SUBMISSION model with extended training")
    print(f"📊 Configuration: LR={config['lr']}, Dropout={config['dropout_rate']}, Noise={config['noise_level']}")
    print(f"⏱️  Extended training: {config['epochs']} epochs, patience={config['early_stopping_patience']}")
    
    # Initialize
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    from src.utils import load_ctf_2025
    (X_profiling, X_attack), (Y_profiling, Y_attack), (P_profiling, P_attack), attack_key = load_ctf_2025(
        filename='../dataset/CHES_2025/CHES_Challenge.h5',
        leakage_model=config['leakage'],
        train_end=config['train_end'],
        poi_start=config['poi_start'], 
        poi_end=config['poi_end']
    )
    
    # Create dataset object to match expected interface
    class DatasetObj:
        def __init__(self, X_profiling, X_attack, Y_profiling, Y_attack, plaintext, key):
            self.X_profiling = X_profiling
            self.X_attack = X_attack
            self.Y_profiling = Y_profiling
            self.Y_attack = Y_attack
            self.plaintext = plaintext
            self.key = key
    
    dataset_obj = DatasetObj(X_profiling, X_attack, Y_profiling, Y_attack, P_attack, attack_key)
    
    # Calculate SNR and select top features
    snr_values = calculate_snr(dataset_obj.X_profiling, dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr_values)[::-1][:config['num_poi']]
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]
    
    print(f"📈 Using full {len(dataset_obj.X_profiling)} profiling traces")
    
    # Single train/validation split (80/20)
    train_index, val_index = train_test_split(
        range(len(dataset_obj.X_profiling)), 
        test_size=0.2, 
        stratify=dataset_obj.Y_profiling, 
        random_state=SEED
    )
    
    print(f"📊 Using single 80/20 train/validation split")
    print(f"🏋️  Training samples: {len(train_index)}")
    print(f"✅ Validation samples: {len(val_index)}")
    
    # Create scaler
    scaler = StandardScaler()
    
    # Create datasets
    train_transform = transforms.Compose([
        DataAugmentation(max_shift=config['max_shift'], noise_level=config['noise_level']),
        ToTensor_trace()
    ])
    eval_transform = transforms.Compose([ToTensor_trace()])
    
    # Get training and validation data
    X_train = dataset_obj.X_profiling[train_index]
    X_val = dataset_obj.X_profiling[val_index]
    Y_train = dataset_obj.Y_profiling[train_index]
    Y_val = dataset_obj.Y_profiling[val_index]
    
    # Fit scaler and transform data
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Create dataloaders
    train_dataset = Custom_Dataset(root='../', dataset=config['dataset'], leakage=config['leakage'])
    train_dataset.X_profiling, train_dataset.Y_profiling = X_train, Y_train
    train_dataset.transform = train_transform
    train_dataset.choose_phase("train")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    
    val_dataset = Custom_Dataset(root='../', dataset=config['dataset'], leakage=config['leakage'])
    val_dataset.X_profiling, val_dataset.Y_profiling = X_val, Y_val
    val_dataset.transform = eval_transform
    val_dataset.choose_phase("train")
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Initialize model
    print(f"🏗️  Initializing CNN model...")
    search_space = {
        'conv_layers': config['conv_layers'],
        'filters': config['filters'],
        'kernels': config['kernels'],
        'pooling_types': config['pooling_types'],
        'pooling_sizes': config['pooling_sizes'],
        'padding': config['padding'],
        'layers': config['layers'],
        'neurons': config['neurons'],
        'activation': config['activation'],
        'dropout_rate': config['dropout_rate']
    }
    model = CNN(search_space, config['num_poi'], 256).to(device)
    
    # Apply weight initialization
    model.apply(weight_init)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,}")
    
    # Create optimizer
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Prepare attack data (transform with scaler)
    X_attack_scaled = scaler.transform(dataset_obj.X_attack)
    
    # Create leakage function for attack evaluation
    def leakage_fn(p, k): 
        return AES_Sbox[k ^ int(p)] if config['leakage'] == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]
    
    # Initialize wandb run (required for attack_driven_training_loop)
    import wandb
    run = wandb.init()
    
    # Train model with attack-driven training
    print(f"🚀 Starting FINAL training...")
    
    model = attack_driven_training_loop(
        config, model, train_loader, device, run,
        X_attack_scaled, dataset_obj.plaintext, 
        dataset_obj.key, leakage_fn
    )
    
    
    # Load the best model
    model_path = f"best_model_fold_0.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"✅ Loaded best model from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load best model, using final model instead: {e}")
    
    # Evaluate the model on the attack set
    from src.utils import evaluate
    print(f"🔍 Evaluating final model...")
    
    GE, NTGE, final_ge = evaluate(device, model, X_attack_scaled, dataset_obj.plaintext, dataset_obj.key,
                                  leakage_fn=leakage_fn, nb_attacks=100,
                                  total_nb_traces_attacks=10000,
                                  nb_traces_attacks=10000)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Final GE: {final_ge}")
    print(f"   NTGE: {NTGE}")
    
    # Save final model
    model_filename = f"submission_model_final_ge_{final_ge}_ntge_{NTGE}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'scaler': scaler,
        'top_k_indices': top_k_indices,
        'final_ge': final_ge,
        'ntge': NTGE
    }, model_filename)
    
    print(f"💾 Final model saved: {model_filename}")
    
    if final_ge == 0:
        print(f"🎉 SUCCESS! Model achieved GE=0 with NTGE={NTGE}")
        print(f"🚀 Ready for competition submission!")
    else:
        print(f"⚠️  Model did not achieve GE=0 (GE={final_ge})")
        print(f"🔄 Consider using refined sweep for better results")
    
    return model_filename, final_ge, NTGE

if __name__ == "__main__":
    train_best_model()
