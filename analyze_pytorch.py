import os
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import CNN
from src.utils import evaluate, AES_Sbox, calculate_HW

if __name__=="__main__":
    dataset = "CHES_2025"
    leakage = "ID"
    nb_traces_attacks = 100000
    total_nb_traces_attacks = 100000

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
    dataloadertest = Custom_Dataset(root='./../', dataset=dataset, leakage="ID",
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

    ##TODO: Load your model ########################
    run_id = "12"
    
    model_filename = f"ge0_model_run_{run_id}.pth"
    config_filename = f"ge0_config_run_{run_id}.npy"
    metadata_filename = f"ge0_metadata_run_{run_id}.json"
    
    print(f"Loading model: {model_filename}")
    print(f"Loading config: {config_filename}")
    print(f"Loading metadata: {metadata_filename}")

    config = np.load(config_filename, allow_pickle=True).item()

    # Manually parse the metadata file to get POIs without importing json
    try:
        with open(metadata_filename, 'r') as f:
            content = f.read()
        
        # Find the start of the POI list
        poi_key = '"poi_indices":'
        key_start_index = content.find(poi_key)
        if key_start_index == -1:
            raise ValueError("'poi_indices' key not found in metadata file.")
            
        list_start_index = content.find('[', key_start_index)
        list_end_index = content.find(']', list_start_index)
        
        # Extract the string of numbers
        poi_string = content[list_start_index + 1 : list_end_index]
        
        # Convert the string of numbers into a list of integers
        poi_indices = [int(s.strip()) for s in poi_string.split(',')]
        
        print(f"Applying {len(poi_indices)} POI indices manually parsed from metadata file.")
        X_attack = X_attack[:, poi_indices]

    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {metadata_filename}")
    except Exception as e:
        print(f"Error during manual parsing of metadata file: {e}")
        raise

    print(f"Data shape after POI selection: {X_attack.shape}")

    #Apply StandardScaler
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_attack = scaler.fit_transform(X_attack)
    print(f"Data shape after scaling: {X_attack.shape}")

    poi_width = X_attack.shape[1] 
    model = CNN(config, poi_width, classes).to(device)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    print("Model loaded successfully.")
    ###############################################################################

    ####All model will be evaluated based on this function ##################
    GE, NTGE, final_ge = evaluate(device, model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn, nb_attacks=100,
                        total_nb_traces_attacks=100000, nb_traces_attacks=100000)
    
    print(f"--- Final Results ---")
    print(f"GE: {GE}")
    print(f"NTGE: {NTGE}")
    print(f"Final GE: {final_ge}")