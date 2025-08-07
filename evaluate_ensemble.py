#!/usr/bin/env python3
"""
Ensemble evaluation for better generalization to private attack traces
"""
import os, numpy as np, torch
from sklearn.preprocessing import StandardScaler
from src.dataloader import Custom_Dataset
from src.net import CNN
from src.utils import evaluate, AES_Sbox
import glob
import json

def load_robust_models():
    """Load all robust GE=0 models for ensemble"""
    model_files = glob.glob("ge0_robust_model_run_*.pth")
    metadata_files = glob.glob("ge0_robust_metadata_run_*.json")
    
    models_info = []
    for model_file in model_files:
        # Find corresponding metadata
        base_name = model_file.replace("ge0_robust_model_", "").replace(".pth", "")
        metadata_file = f"ge0_robust_metadata_{base_name}.json"
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            models_info.append({
                'model_file': model_file,
                'metadata': metadata,
                'ntge': metadata['NTGE'],
                'robustness_penalty': metadata.get('robustness_penalty', 999)
            })
    
    # Sort by robustness (lower penalty is better)
    models_info.sort(key=lambda x: (x['robustness_penalty'], x['ntge']))
    
    return models_info

def ensemble_predict(models, X_attack, plt_attack, correct_key, device, leakage_fn):
    """Ensemble prediction from multiple models"""
    all_predictions = []
    
    for model_info in models:
        model_file = model_info['model_file']
        metadata = model_info['metadata']
        
        # Load model architecture
        poi_width = metadata['num_poi']
        classes = 256 if metadata['leakage'] == 'ID' else 9
        
        # Create model with same architecture
        search_space = {
            'conv_layers': 2,
            'filters': 16,
            'kernels': 36,
            'dropout_rate': 0.2,
            'activation': 'selu'
        }
        
        model = CNN(search_space, poi_width, classes).to(device)
        
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            
            # Get predictions from this model
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_attack).to(device)
                predictions = model(X_tensor).cpu().numpy()
                all_predictions.append(predictions)
                
            print(f"✅ Loaded model: {model_file} (NTGE={metadata['NTGE']}, Robust={metadata.get('robustness_penalty', 'N/A')})")
            
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")
            continue
    
    if not all_predictions:
        print("❌ No models loaded successfully!")
        return None, None, None
    
    # Ensemble averaging
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Evaluate ensemble
    print(f"\n🔗 Ensemble evaluation with {len(all_predictions)} models:")
    
    # Convert to rank format for evaluation
    ranks = []
    for i in range(len(X_attack)):
        true_key = correct_key
        pred_probs = ensemble_predictions[i]
        
        # Calculate probabilities for each key guess
        key_scores = []
        for key_guess in range(256):
            intermediate = leakage_fn(plt_attack[i], key_guess)
            score = pred_probs[intermediate]
            key_scores.append(score)
        
        # Get rank of correct key
        key_scores = np.array(key_scores)
        rank = np.where(np.argsort(key_scores)[::-1] == true_key)[0][0]
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # Calculate metrics
    ge_curve = []
    for num_traces in range(1, len(ranks) + 1):
        ge = np.mean(np.minimum.accumulate(ranks[:num_traces]))
        ge_curve.append(ge)
    
    ntge = np.argmax(np.array(ge_curve) == 0) + 1 if 0 in ge_curve else len(ge_curve)
    final_ge = ge_curve[-1]
    
    return ge_curve, ntge, final_ge

def evaluate_ensemble_robustness():
    """Evaluate ensemble on multiple noise levels"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset_obj = Custom_Dataset(root='../', dataset='CHES_2025', leakage='ID',
                                poi_start=0, poi_end=7000, train_end=0, test_end=100000)
    
    # Load POI metadata
    try:
        poi_metadata = np.load('poi_selection_metadata.npy', allow_pickle=True).item()
        top_k_indices = np.array(poi_metadata['poi_indices'])
        print(f"📍 Using saved POI selection: {len(top_k_indices)} points")
    except:
        print("❌ POI metadata not found - using default SNR selection")
        from src.utils import calculate_snr
        snr = calculate_snr(dataset_obj.X_profiling[:10000], dataset_obj.Y_profiling[:10000])  # Sample for speed
        top_k_indices = np.argsort(snr)[-150:]  # Default fallback
    
    # Apply POI selection
    X_attack = dataset_obj.X_attack[:, top_k_indices]
    
    # Standard scaling (approximate - ideally use saved scaler)
    scaler = StandardScaler()
    X_attack_scaled = scaler.fit_transform(X_attack)
    
    # Load robust models
    models_info = load_robust_models()
    
    if not models_info:
        print("❌ No robust models found!")
        return
    
    print(f"🔗 Found {len(models_info)} robust models for ensemble")
    
    # Select top 3-5 most robust models
    top_models = models_info[:min(5, len(models_info))]
    
    def leakage_fn(p, k): 
        return AES_Sbox[k ^ int(p)]
    
    # Test on different noise levels
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    
    for noise_level in noise_levels:
        print(f"\n🔊 Testing noise level: σ={noise_level}")
        
        # Add noise
        if noise_level > 0:
            X_test = X_attack_scaled + np.random.normal(0, noise_level, X_attack_scaled.shape)
        else:
            X_test = X_attack_scaled
        
        # Ensemble evaluation
        ge_curve, ntge, final_ge = ensemble_predict(
            top_models, X_test, dataset_obj.plt_attack, dataset_obj.correct_key, 
            device, leakage_fn
        )
        
        if ge_curve is not None:
            print(f"📊 Ensemble Performance (σ={noise_level}): GE={final_ge:.1f}, NTGE={ntge}")
        else:
            print(f"❌ Ensemble evaluation failed for σ={noise_level}")

if __name__ == "__main__":
    evaluate_ensemble_robustness()
