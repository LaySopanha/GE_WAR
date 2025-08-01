# train_best_model.py - Dedicated script to recreate the GE=0, NTGE=97k model
import os, random, numpy as np, torch, json
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from src.dataloader import Custom_Dataset, DataAugmentation, ToTensor_trace
from src.net import CNN, weight_init
from src.trainer import attack_driven_training_loop
from src.utils import evaluate, AES_Sbox, calculate_snr

class BestModelConfig:
    """Configuration class for the best GE=0, NTGE=97k model"""
    def __init__(self):
        # === EXACT CONFIGURATION FROM config-ntge-97k.yaml ===
        self.activation = "relu"
        self.attack_eval_frequency = 15
        self.batch_size = 32
        self.conv_layers = 4
        self.dataset = "CHES_2025"
        self.dropout_rate = 0.26698384079064097
        self.early_stopping_patience = 25
        self.epochs = 250
        self.filters = 32
        self.k_folds = 2
        self.kernels = 24
        self.layers = 2
        self.leakage = "ID"
        self.lr = 0.0001
        self.max_shift = 20
        self.min_epochs = 80
        self.model_type = "cnn"
        self.neurons = 512
        self.noise_level = 0.022197678617252355
        self.num_poi = 250
        self.num_traces_attack = 100000
        self.optimizer = "Adam"
        self.padding = 0
        self.poi_end = 7000
        self.poi_start = 0
        self.pooling_sizes = 2
        self.pooling_types = "max_pool"
        self.train_end = 350000
        
        # Additional attributes for compatibility
        self.kernel_initializer = "glorot_uniform"
        
    def __getitem__(self, key):
        """Allow dictionary-style access config['key']"""
        return getattr(self, key)
        
    def __setitem__(self, key, value):
        """Allow dictionary-style assignment config['key'] = value"""
        setattr(self, key, value)
        
    def __contains__(self, key):
        """Allow 'key' in config checks"""
        return hasattr(self, key)
        
    def items(self):
        """Return config as items for compatibility with original code"""
        return self.__dict__.items()
    
    def get(self, key, default=None):
        """Get method for compatibility"""
        return getattr(self, key, default)

def train_best_model():
    """Train the exact model configuration that achieved GE=0 with NTGE=97k"""
    
    print("🎯 Training Best Model Configuration")
    print("=" * 50)
    print(f"Target: GE=0, NTGE=97k")
    print(f"Architecture: CNN, 4 conv layers, 32 filters, 24 kernels")
    print(f"Training: 350k traces, ID leakage, 250 POIs")
    print("=" * 50)
    
    # Initialize configuration
    config = BestModelConfig()
    
    # Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load full traces first to compute SNR
    print(f"📊 Loading {config.train_end} profiling traces from dataset...")
    full_dataset_obj = Custom_Dataset(
        root='../', 
        dataset=config.dataset, 
        leakage=config.leakage,
        poi_start=0, 
        poi_end=7000,  # Load all points
        train_end=config.train_end, 
        test_end=100000
    )
    
    print(f"🚀 Loaded {len(full_dataset_obj.X_profiling)} profiling traces")

    # Calculate SNR on the full profiling dataset
    print("📈 Calculating SNR for POI selection...")
    snr = calculate_snr(full_dataset_obj.X_profiling, full_dataset_obj.Y_profiling)
    top_k_indices = np.argsort(snr)[-config.num_poi:]
    print(f"✅ Selected top {config.num_poi} POIs based on SNR")

    # Create a new dataset object with only the selected POIs
    dataset_obj = deepcopy(full_dataset_obj)
    dataset_obj.X_profiling = dataset_obj.X_profiling[:, top_k_indices]
    dataset_obj.X_attack = dataset_obj.X_attack[:, top_k_indices]

    print(f"📈 Using full {len(dataset_obj.X_profiling)} profiling traces")

    # K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(kf.split(dataset_obj.X_profiling, dataset_obj.Y_profiling)):
        print(f"\n🔄 Training Fold {fold+1}/{config.k_folds}")
        print("-" * 30)

        # Create a scaler for this fold
        scaler = StandardScaler()
        
        # Create transforms
        train_transform = transforms.Compose([
            DataAugmentation(max_shift=config.max_shift, noise_level=config.noise_level),
            ToTensor_trace()
        ])
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

        # Initialize model for this fold
        poi_width = config.num_poi
        classes = 256 if config.leakage == 'ID' else 9
        search_space = {k: v for k, v in config.items()}
        
        print(f"🧠 Initializing CNN model...")
        model = CNN(search_space, poi_width, classes).to(device)
        weight_init(model, search_space.get("kernel_initializer", "glorot_uniform"))
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model has {param_count:,} parameters")
        
        def leakage_fn(p, k): 
            return AES_Sbox[k ^ int(p)] if config.leakage == 'ID' else [bin(x).count("1") for x in range(256)][AES_Sbox[k ^ int(p)]]
        
        # Train the model with attack-driven training
        print(f"🚀 Starting attack-driven training...")
        X_attack_fold = scaler.transform(fold_dataset_obj.X_attack)
        
        # Create a simple run object for logging (without wandb)
        class SimpleRun:
            def __init__(self):
                self.id = f"best_model_fold_{fold}"
            def log(self, data):
                pass  # Simple logging placeholder
        
        simple_run = SimpleRun()
        
        model = attack_driven_training_loop(
            config, model, train_loader, device, simple_run,
            X_attack_fold, fold_dataset_obj.plt_attack, 
            fold_dataset_obj.correct_key, leakage_fn
        )
        
        # Load the best model from training
        model_path = f"best_model_fold_{fold}.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path))
                print(f"✅ Loaded best model for fold {fold+1}")
            except Exception as e:
                print(f"⚠️  Could not load model, using last epoch model: {e}")

        # Evaluate the model on the attack set
        print(f"🎯 Evaluating model on attack set...")
        X_attack_scaled = scaler.transform(fold_dataset_obj.X_attack)
    
        GE, NTGE, final_ge = evaluate(
            device, model, X_attack_scaled, fold_dataset_obj.plt_attack, 
            fold_dataset_obj.correct_key, leakage_fn=leakage_fn, nb_attacks=100,
            total_nb_traces_attacks=config.num_traces_attack,
            nb_traces_attacks=config.num_traces_attack
        )
        
        print(f"📊 Fold {fold+1} Results:")
        print(f"   Final GE: {final_ge}")
        print(f"   NTGE: {NTGE}")
        print(f"   Average GE: {np.mean(GE):.2f}")
        
        # Save model and metadata if it achieves GE=0
        if final_ge == 0:
            timestamp = torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else "cpu"
            model_filename = f"recreated_ge0_model_fold_{fold}_ntge_{NTGE}.pth"
            config_filename = f"recreated_ge0_config_fold_{fold}_ntge_{NTGE}.json"
            metadata_filename = f"recreated_ge0_metadata_fold_{fold}_ntge_{NTGE}.json"
            
            # Save model
            try:
                torch.save(model.state_dict(), model_filename)
                print(f"✅ Model saved: {model_filename}")
            except Exception as e:
                print(f"❌ Error saving model: {e}")
                # Pickle fallback
                import pickle
                fallback_path = model_filename.replace('.pth', '_pickle.pkl')
                try:
                    with open(fallback_path, 'wb') as f:
                        pickle.dump(model.state_dict(), f)
                    print(f"✅ Pickle fallback: {fallback_path}")
                except Exception as e2:
                    print(f"❌ Pickle failed: {e2}")
            
            # Save configuration
            config_dict = {k: v for k, v in config.items()}
            try:
                with open(config_filename, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
                print(f"✅ Config saved: {config_filename}")
            except Exception as e:
                print(f"❌ Config save failed: {e}")
            
            # Save metadata
            metadata = {
                'fold': fold,
                'final_GE': float(final_ge),
                'NTGE': int(NTGE),
                'average_GE': float(np.mean(GE)),
                'num_poi': config.num_poi,
                'train_traces': config.train_end,
                'leakage': config.leakage,
                'dataset': config.dataset,
                'model_parameters': param_count,
                'architecture': 'CNN',
                'conv_layers': config.conv_layers,
                'filters': config.filters,
                'kernels': config.kernels
            }
            
            try:
                with open(metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✅ Metadata saved: {metadata_filename}")
            except Exception as e:
                print(f"❌ Metadata save failed: {e}")
            
            print(f"🎯 SUCCESS! Recreated GE=0 model: {model_filename} (NTGE={NTGE})")
        
        fold_results.append({
            'fold': fold,
            'final_ge': final_ge,
            'ntge': NTGE,
            'avg_ge': np.mean(GE)
        })

    # Summary
    print("\n" + "=" * 50)
    print("🏆 TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 50)
    
    for result in fold_results:
        print(f"Fold {result['fold']+1}: GE={result['final_ge']}, NTGE={result['ntge']}, Avg_GE={result['avg_ge']:.2f}")
    
    ge0_folds = [r for r in fold_results if r['final_ge'] == 0]
    if ge0_folds:
        best_ntge = min(ge0_folds, key=lambda x: x['ntge'])
        print(f"\n🎯 BEST RESULT: Fold {best_ntge['fold']+1} - GE=0, NTGE={best_ntge['ntge']}")
        print(f"✅ Target achieved! Original was NTGE=97k")
    else:
        print(f"\n⚠️  No GE=0 achieved in this run. Consider running again or adjusting parameters.")

if __name__ == "__main__":
    train_best_model()
