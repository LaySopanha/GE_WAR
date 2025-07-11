import os
import random
from copy import deepcopy
import numpy as np
import torch

from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN
from src.trainer import trainer
# Import all your utility functions
from src.utils import AES_Sbox, calculate_HW, normalize_trace, align_traces
from torch.utils.tensorboard import SummaryWriter


if __name__=="__main__":
    dataset = "CHES_2025"
    model_type = "cnn"
    leakage = "HW"
    train_models = True
    num_epochs = 50
    total_num_models = 20

    # Directory setup
    if not os.path.exists('./Result/'):
        os.mkdir('./Result/')
    root = "./Result/"
    save_root = root + dataset + "_" + model_type + "_" + leakage + "/"
    model_root = save_root + "models/"
    print("root:", root)
    print("save_time_path:", save_root)
    if not os.path.exists(root): os.mkdir(root)
    if not os.path.exists(save_root): os.mkdir(save_root)
    if not os.path.exists(model_root): os.mkdir(model_root)

    # Seeding for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Leakage model setup (vectorized for speed)
    if leakage == 'ID':
        def leakage_fn(att_plt, k):
            return AES_Sbox[k ^ int(att_plt)]
        classes = 256
    elif leakage == 'HW':
        hw = np.array([bin(x).count("1") for x in range(256)])
        sbox = np.array(AES_Sbox)
        def leakage_fn(att_plt, k):
            plaintexts_as_int = att_plt.astype(np.uint8)
            sbox_in = np.bitwise_xor(plaintexts_as_int, k)
            sbox_out = sbox[sbox_in]
            return hw[sbox_out]
        classes = 9

    # Data Loading (loads the raw data)
    dataloadertrain = Custom_Dataset(root='./../', dataset=dataset, leakage="ID",
                                                 transform=transforms.Compose([ToTensor_trace()]))

    # --- FULL PREPROCESSING PIPELINE ---
    
    # 1. Create a reference trace for alignment
    print("Creating reference trace for alignment...")
    reference_trace = np.mean(dataloadertrain.X_profiling[:1000], axis=0)
    
    # 2. Define alignment hyperparameter
    max_shift = 50 # This is a key parameter to tune!
    
    # 3. Apply Trace Alignment (Step 1 of preprocessing)
    print("Applying data alignment...")
    dataloadertrain.X_profiling = align_traces(dataloadertrain.X_profiling, reference_trace, max_shift)
    dataloadertrain.X_attack = align_traces(dataloadertrain.X_attack, reference_trace, max_shift)
    print("Alignment complete.")

    # 4. Apply Normalization (Step 2 of preprocessing)
    print("Applying data normalization...")
    dataloadertrain.X_profiling = normalize_trace(dataloadertrain.X_profiling)
    dataloadertrain.X_attack = normalize_trace(dataloadertrain.X_attack)
    print("Data normalization complete.")

    # --- END OF PREPROCESSING ---

    # Label Conversion
    if leakage == "HW":
        dataloadertrain.Y_profiling = np.array(calculate_HW(dataloadertrain.Y_profiling))
        dataloadertrain.Y_attack = np.array(calculate_HW(dataloadertrain.Y_attack))

    # Data Splitting
    dataloadertrain.split_attack_set_validation_test()
    dataloadertrain.choose_phase("train")
    dataloaderval = deepcopy(dataloadertrain)
    dataloaderval.choose_phase("validation")
    
    # Prepare data for validation key rank check
    # Note: X_val and plt_val now come from the ALIGNED and NORMALIZED X_attack set
    X_val = dataloaderval.X_attack_val
    plt_val = dataloaderval.plt_attack # Assuming plt_attack corresponds to X_attack
    # This might need adjustment if your split function also splits plaintexts
    # Let's assume you need to split plt_attack similarly
    from sklearn.model_selection import train_test_split # If needed
    _, plt_val_split = train_test_split(dataloadertrain.plt_attack, test_size=0.1, random_state=0)
    plt_val = plt_val_split

    correct_key = dataloadertrain.correct_key
    num_sample_pts = dataloadertrain.X_profiling.shape[1]
    
    # Random Search Loop
    for num_models in range(total_num_models):
        if train_models == True:
            config = create_hyperparameter_space(model_type)
            np.save(model_root + "model_configuration_"+str(num_models)+".npy", config)

            # TensorBoard Setup
            log_dir = os.path.join(save_root, "logs", f"run_{num_models}")
            writer = SummaryWriter(log_dir=log_dir)
            print(f"\n--- Starting training for Model {num_models}. Logging to: {log_dir} ---")
            
            # DataLoaders Setup
            batch_size = config["batch_size"]
            dataloaders = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size, shuffle=True),
                           "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size, shuffle=True)}
            dataset_sizes = {"train": len(dataloadertrain), "val": len(dataloaderval)}

            # Run Trainer
            model, final_metrics = trainer(
                config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, 
                model_type, classes, device, writer, 
                X_val, plt_val, leakage_fn, correct_key
            )
            
            # Save Model
            torch.save(model.state_dict(), model_root + "model_"+str(num_models)+".pth")

            # Log to TensorBoard HPARAMS
            writer.add_hparams(
                hparam_dict=config,
                metric_dict={
                    'hparam/final_val_loss': final_metrics.get('val_loss', float('inf')),
                    'hparam/final_val_acc': final_metrics.get('val_acc', 0.0),
                    'hparam/final_val_rank': final_metrics.get('val_rank', float('inf'))
                }
            )

            writer.close()
            
        print(f"--- Finished training Model {num_models}. Model and config saved. ---")