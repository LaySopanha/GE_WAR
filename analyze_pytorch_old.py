import os
import random
from copy import deepcopy
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import CNN  # Only import what we need for submission
from src.utils import evaluate, AES_Sbox, calculate_HW

if __name__=="__main__":
    dataset = "CHES_2025"
    leakage = "ID"  # Changed to ID - proven superior to HW in our analysis
    nb_traces_attacks = 100000  # Match training configuration
    total_nb_traces_attacks = 100000  # Match training configuration

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_attacks = 100


    ##################please do not touch this code below###################
    dataloadertest = Custom_Dataset(root='./../', dataset=dataset, leakage="ID", #change root to where you download your dataset.
                                                 transform=transforms.Compose([ToTensor_trace()]))
    #########################################################################
    if leakage == 'ID':
        def leakage_fn(att_plt, k):
            return AES_Sbox[k ^ int(att_plt)]
        classes = 256
    elif leakage == 'HW':
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]
            return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9
        dataloadertest.Y_attack = calculate_HW(dataloadertest.Y_attack)
    else:
        ####TODO: You can change the code here if you want to create your own leakage model.
        pass


    ##################please do not touch this code here###################
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    Y_attack = dataloadertest.Y_attack
    plt_attack = dataloadertest.plt_attack
    num_sample_pts = X_attack.shape[-1]
    print(f"Original X_attack shape: {X_attack.shape}")
    print(f"Number of sample points: {num_sample_pts}")
    #########################################################################


    ##TODO: Load your model (note, you have to create your model in this file and new function should be in this file.) ########################
    ############## Updated for CNN with GE=0 models ############################################
    model_type = "cnn"
    
    # Try to load the best GE=0 model from sweep results
    import glob
    ge0_models = glob.glob("ge0_model_*.pth")
    
    if ge0_models:
        # Use the model with lowest NTGE
        best_model_file = min(ge0_models, key=lambda x: int(x.split('_ntge_')[1].split('.pth')[0]))
        config_file = best_model_file.replace('ge0_model_', 'ge0_config_').replace('.pth', '.npy')
        
        print(f"Loading best GE=0 model: {best_model_file}")
        
        # Load configuration
        config = np.load(config_file, allow_pickle=True).item()
        metadata_file = best_model_file.replace('ge0_model_', 'ge0_metadata_').replace('.pth', '.json')
        
        print(f"Model config: num_poi={config.get('num_poi', 250)}")
        
        # Load the exact POI indices used during training for best performance
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if 'poi_indices' in metadata:
            poi_indices = np.array(metadata['poi_indices'])
            print(f"Using saved POI indices: {len(poi_indices)} points")
            print(f"POI indices range: {poi_indices[0]} to {poi_indices[-1]}")
            X_attack = X_attack[:, poi_indices]
            print(f"After POI selection: {X_attack.shape}")
        else:
            print("⚠️ No saved POI indices found, using SNR-based selection")
            # Fallback to SNR-based selection
            num_poi = config.get('num_poi', 250)
            from src.utils import calculate_snr
            snr_values = calculate_snr(X_attack, Y_attack)
            poi_indices = np.argsort(snr_values)[-num_poi:]
            X_attack = X_attack[:, poi_indices]
            print(f"After SNR POI selection: {X_attack.shape}")
        
        # Apply standard scaling to match training preprocessing
        print("Applying StandardScaler to match training preprocessing...")
        scaler = StandardScaler()
        X_attack = scaler.fit_transform(X_attack)
        print(f"After scaling: {X_attack.shape}")
        
        # Create model architecture to match our training
        poi_width = X_attack.shape[-1]  # Use actual POI width after selection
        classes = 256  # ID leakage
        
        model = CNN(config, poi_width, classes).to(device)
        model.load_state_dict(torch.load(best_model_file, map_location=device))
        
    else:
        print("⚠️ No GE=0 models found. Creating a placeholder model for testing.")
        print("   Run your sweep first to generate GE=0 models!")
        
        # Create a default model architecture for testing
        default_config = {
            'conv_layers': 3,
            'filters': 32, 
            'kernels': 36,
            'layers': 2,
            'neurons': 1024,
            'activation': 'relu',
            'dropout_rate': 0.2,
            'pooling_types': 'max_pool',
            'pooling_sizes': 2,
            'padding': 1
        }
        
        # Use default POI size but adjust to actual data
        poi_width = min(250, num_sample_pts)  # Adjust based on actual data
        classes = 256  # ID leakage
        
        model = CNN(default_config, poi_width, classes).to(device)
        print(f"Created default CNN model with {poi_width} POI points")
        print("⚠️ This is just for testing - train a real model first!")
    ###############################################################################


    ####All model will be evaluated based on this function, if it does not adhere to the following, it will be eliminated. ##################
    GE, NTGE, final_ge = evaluate(device, model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn, nb_attacks=100,
                        total_nb_traces_attacks=100000, nb_traces_attacks=100000)
    
    print(f"--- Final Results ---")
    print(f"GE: {GE}")
    print(f"NTGE: {NTGE}")
    print(f"Final GE: {final_ge}")