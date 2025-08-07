#!/usr/bin/env python3
"""
Select the best single model for competition submission
"""
import os
import numpy as np
import torch
import json
import glob
from sklearn.preprocessing import StandardScaler
from src.dataloader import Custom_Dataset
from src.net import CNN
from src.utils import evaluate, AES_Sbox

def evaluate_single_models():
    """Evaluate all trained models and select the best one"""
    
    # Find all robust models
    model_files = glob.glob("ge0_robust_model_run_*.pth")
    
    if not model_files:
        print("❌ No robust models found!")
        return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset_obj = Custom_Dataset(root='../', dataset='CHES_2025', leakage='ID',
                                poi_start=0, poi_end=7000, train_end=0, test_end=100000)
    
    # Load POI selection
    poi_metadata = np.load('poi_selection_metadata.npy', allow_pickle=True).item()
    top_k_indices = np.array(poi_metadata['poi_indices'])
    X_attack = dataset_obj.X_attack[:, top_k_indices]
    
    # Scale data
    scaler = StandardScaler()
    X_attack_scaled = scaler.fit_transform(X_attack)
    
    best_model = None
    best_score = float('inf')
    best_info = None
    
    def leakage_fn(p, k): 
        return AES_Sbox[k ^ int(p)]
    
    print("🔍 Evaluating individual models for competition submission...")
    
    for model_file in model_files:
        # Load metadata
        base_name = model_file.replace("ge0_robust_model_", "").replace(".pth", "")
        metadata_file = f"ge0_robust_metadata_{base_name}.json"
        
        if not os.path.exists(metadata_file):
            continue
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        poi_width = metadata['num_poi']
        classes = 256 if metadata['leakage'] == 'ID' else 9
        
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
            
            # Evaluate on clean data
            GE_clean, NTGE_clean, final_ge_clean = evaluate(
                device, model, X_attack_scaled, dataset_obj.plt_attack, 
                dataset_obj.correct_key, leakage_fn=leakage_fn, nb_attacks=100,
                total_nb_traces_attacks=100000, nb_traces_attacks=100000
            )
            
            # Evaluate with noise (robustness test)
            X_noisy = X_attack_scaled + np.random.normal(0, 0.02, X_attack_scaled.shape)
            GE_noisy, NTGE_noisy, final_ge_noisy = evaluate(
                device, model, X_noisy, dataset_obj.plt_attack, 
                dataset_obj.correct_key, leakage_fn=leakage_fn, nb_attacks=50,
                total_nb_traces_attacks=100000, nb_traces_attacks=100000
            )
            
            # Competition score: prioritize GE=0, then NTGE, then robustness
            robustness_penalty = final_ge_noisy - final_ge_clean
            competition_score = (
                1000000 * final_ge_clean +  # Must be 0 for good score
                NTGE_clean +                # Lower is better
                10 * robustness_penalty     # Penalty for poor robustness
            )
            
            print(f"📊 {model_file}")
            print(f"    Clean: GE={final_ge_clean}, NTGE={NTGE_clean}")
            print(f"    Noisy: GE={final_ge_noisy}, Robustness penalty={robustness_penalty:.1f}")
            print(f"    Competition score: {competition_score:.1f}")
            
            if competition_score < best_score:
                best_score = competition_score
                best_model = model_file
                best_info = {
                    'model_file': model_file,
                    'metadata': metadata,
                    'clean_ge': final_ge_clean,
                    'clean_ntge': NTGE_clean,
                    'noisy_ge': final_ge_noisy,
                    'robustness_penalty': robustness_penalty,
                    'competition_score': competition_score
                }
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_file}: {e}")
            continue
    
    if best_model:
        print(f"\n🏆 BEST MODEL FOR COMPETITION:")
        print(f"📁 File: {best_info['model_file']}")
        print(f"🎯 Clean performance: GE={best_info['clean_ge']}, NTGE={best_info['clean_ntge']}")
        print(f"🛡️ Robustness penalty: {best_info['robustness_penalty']:.1f}")
        print(f"📊 Competition score: {best_info['competition_score']:.1f}")
        
        # Copy best model to submission file
        import shutil
        shutil.copy2(best_model, "competition_best_single_model.pth")
        
        # Save submission metadata
        np.save("competition_best_single_metadata.npy", best_info)
        
        print(f"✅ Submission ready: competition_best_single_model.pth")
        
    else:
        print("❌ No suitable models found!")

if __name__ == "__main__":
    evaluate_single_models()
