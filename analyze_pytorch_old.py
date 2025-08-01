import os
import random
from copy import deepcopy
import numpy as np
import torch

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
        
        # Create model architecture to match our training
        poi_width = config.get('num_poi', 250)  # Default POI if not in config
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
    GE, NTGE = evaluate(device, model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn, nb_attacks=100,
                        total_nb_traces_attacks=100000, nb_traces_attacks=100000)