# generalization_test.py - Test model robustness across different validation scenarios
import os, random, numpy as np, torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
from src.trainer import attack_driven_training_loop
from src.utils import calculate_snr

def test_generalization():
    """Test model generalization with multiple validation strategies"""
    
    # Best model config from your sweep
    config = {
        'model_type': 'cnn',
        'dropout_rate': 0.2548776576936158,
        'layers': 2, 'neurons': 512, 'activation': 'relu',
        'conv_layers': 4, 'filters': 32, 'kernels': 24,
        'pooling_types': 'max_pool', 'pooling_sizes': 2, 'padding': 0,
        'early_stopping_patience': 25, 'min_epochs': 60,
        'attack_eval_frequency': 20, 'epochs': 300,
        'lr': 0.0001, 'optimizer': 'Adam', 'batch_size': 32,
        'max_shift': 20, 'noise_level': 0.01969817242376655,
        'dataset': 'CHES_2025', 'leakage': 'ID', 'num_poi': 250,
        'poi_start': 0, 'poi_end': 7000, 'train_end': 350000,
        'num_traces_attack': 100000
    }
    
    print("🧪 GENERALIZATION ROBUSTNESS TEST")
    print("="*60)
    
    # Initialize
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
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
    
    # Calculate SNR and select features
    snr_values = calculate_snr(X_profiling, Y_profiling)
    top_k_indices = np.argsort(snr_values)[::-1][:config['num_poi']]
    X_data = X_profiling[:, top_k_indices]
    Y_data = Y_profiling
    
    results = []
    
    # TEST 1: Multiple Random Seeds (Data Distribution Robustness)
    print("\n📊 TEST 1: Multiple Random Seeds")
    print("-" * 40)
    
    for seed in [42, 123, 456, 789, 999]:
        print(f"Testing with seed {seed}...")
        
        # Split with different seed
        train_idx, val_idx = train_test_split(
            range(len(X_data)), test_size=0.2, 
            stratify=Y_data, random_state=seed
        )
        
        # Quick training (reduced epochs for testing)
        final_ge, ntge = train_and_evaluate(
            X_data, Y_data, train_idx, val_idx, 
            config, device, epochs=150, seed=seed
        )
        
        results.append({
            'test': f'Seed_{seed}',
            'final_ge': final_ge,
            'ntge': ntge
        })
        
        print(f"  Seed {seed}: GE={final_ge}, NTGE={ntge}")
    
    # TEST 2: Different Train/Val Ratios (Dataset Size Robustness)
    print("\n📊 TEST 2: Different Train/Val Split Ratios")
    print("-" * 40)
    
    for test_size in [0.15, 0.20, 0.25, 0.30]:
        print(f"Testing with {int((1-test_size)*100)}/{int(test_size*100)} split...")
        
        train_idx, val_idx = train_test_split(
            range(len(X_data)), test_size=test_size, 
            stratify=Y_data, random_state=42
        )
        
        final_ge, ntge = train_and_evaluate(
            X_data, Y_data, train_idx, val_idx, 
            config, device, epochs=150, seed=42
        )
        
        results.append({
            'test': f'Split_{int((1-test_size)*100)}_{int(test_size*100)}',
            'final_ge': final_ge,
            'ntge': ntge
        })
        
        print(f"  {int((1-test_size)*100)}/{int(test_size*100)}: GE={final_ge}, NTGE={ntge}")
    
    # TEST 3: Different Trace Subsets (Data Distribution Robustness)
    print("\n📊 TEST 3: Different Trace Subsets")
    print("-" * 40)
    
    for start_pct in [0, 10, 20, 30]:
        end_pct = start_pct + 70  # Use 70% of data each time
        start_idx = int(len(X_data) * start_pct / 100)
        end_idx = int(len(X_data) * end_pct / 100)
        
        print(f"Testing with traces {start_pct}%-{end_pct}%...")
        
        subset_X = X_data[start_idx:end_idx]
        subset_Y = Y_data[start_idx:end_idx]
        
        train_idx, val_idx = train_test_split(
            range(len(subset_X)), test_size=0.2, 
            stratify=subset_Y, random_state=42
        )
        
        final_ge, ntge = train_and_evaluate(
            subset_X, subset_Y, train_idx, val_idx, 
            config, device, epochs=150, seed=42
        )
        
        results.append({
            'test': f'Subset_{start_pct}_{end_pct}',
            'final_ge': final_ge,
            'ntge': ntge
        })
        
        print(f"  Traces {start_pct}%-{end_pct}%: GE={final_ge}, NTGE={ntge}")
    
    # ANALYSIS
    print("\n🎯 GENERALIZATION ANALYSIS")
    print("="*60)
    
    ge0_results = [r for r in results if r['final_ge'] == 0]
    all_ntge = [r['ntge'] for r in ge0_results]
    
    if len(ge0_results) > 0:
        mean_ntge = np.mean(all_ntge)
        std_ntge = np.std(all_ntge)
        min_ntge = np.min(all_ntge)
        max_ntge = np.max(all_ntge)
        
        print(f"✅ GE=0 Success Rate: {len(ge0_results)}/{len(results)} ({len(ge0_results)/len(results)*100:.1f}%)")
        print(f"📊 NTGE Statistics:")
        print(f"   Mean: {mean_ntge:.0f}")
        print(f"   Std:  {std_ntge:.0f}")
        print(f"   Range: {min_ntge:.0f} - {max_ntge:.0f}")
        print(f"   CV: {std_ntge/mean_ntge*100:.1f}%")
        
        # Generalization assessment
        cv = std_ntge/mean_ntge*100
        success_rate = len(ge0_results)/len(results)*100
        
        print(f"\n🔮 GENERALIZATION PREDICTION:")
        if cv < 15 and success_rate > 80:
            print("🟢 EXCELLENT: Very likely to generalize well to private datasets")
            expected_range = f"{mean_ntge*0.9:.0f} - {mean_ntge*1.2:.0f}"
        elif cv < 25 and success_rate > 60:
            print("🟡 GOOD: Likely to perform reasonably on private datasets")
            expected_range = f"{mean_ntge*0.8:.0f} - {mean_ntge*1.4:.0f}"
        else:
            print("🔴 CONCERNING: May struggle with different private datasets")
            expected_range = f"{mean_ntge*0.6:.0f} - {mean_ntge*1.8:.0f}"
        
        print(f"📈 Expected Private Dataset NTGE Range: {expected_range}")
        
    else:
        print("❌ WARNING: No models achieved GE=0 in robustness testing!")
        print("🔄 Consider retraining with more robust hyperparameters")
    
    return results

def train_and_evaluate(X_data, Y_data, train_idx, val_idx, config, device, epochs=150, seed=42):
    """Quick training and evaluation for robustness testing"""
    
    # Set seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    # Prepare data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_data[train_idx])
    X_val = scaler.transform(X_data[val_idx])
    Y_train, Y_val = Y_data[train_idx], Y_data[val_idx]
    
    # Create dataloaders
    train_transform = transforms.Compose([
        DataAugmentation(max_shift=config['max_shift'], noise_level=config['noise_level']),
        ToTensor_trace()
    ])
    eval_transform = transforms.Compose([ToTensor_trace()])
    
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
    
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Load dataset for attack evaluation
    from src.utils import load_ctf_2025
    (X_prof, X_att), (Y_prof, Y_att), (P_prof, P_att), key = load_ctf_2025(
        filename='../dataset/CHES_2025/CHES_Challenge.h5',
        leakage_model=config['leakage'],
        train_end=config['train_end'],
        poi_start=config['poi_start'], 
        poi_end=config['poi_end']
    )
    
    # Create a simple object to match expected interface
    class DatasetObj:
        def __init__(self, plaintext, key):
            self.plaintext = plaintext
            self.key = key
    
    dataset_obj = DatasetObj(P_att, key)
    
    # Quick training
    try:
        _, _, final_ge, ntge, _ = attack_driven_training_loop(
            model=model, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            attack_loader=val_loader, plaintext=dataset_obj.plaintext,
            key=dataset_obj.key, device=device, epochs=epochs,
            early_stopping_patience=config['early_stopping_patience'],
            min_epochs=config['min_epochs'],
            attack_eval_frequency=config['attack_eval_frequency'],
            verbose=False
        )
        return final_ge, ntge
    except Exception as e:
        print(f"    Error in training: {e}")
        return 999, 999999

if __name__ == "__main__":
    test_generalization()
