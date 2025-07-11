import numpy as np
import csv
import os 

# -- config ---
# Reuslt directory
results_dir = "./Result/CHES_2025_cnn_HW/models/best_models/"
num_models_to_check = 2


# --- analysis ---
best_model_index = -1
best_ntge = float('inf')
best_final_ge = float('inf')
print("--- Analyzing Scout Run Results ---")

for i in range(num_models_to_check):
    result_path = os.path.join(results_dir, f"result_{i}.npy")
    config_path = os.path.join(results_dir, f"model_configuration_{i}.npy")

    if not os.path.exists(result_path):
        print(f"Warning: Result file for model {i} not found. Skipping.")
        continue
    
    # load the results dictionary
    result = np.load(result_path , allow_pickle=True).item()
    ge_trace = result['GE']
    ntge = result['NTGE']
    
    # The last value of the GE array which is 0
    final_ge = ge_trace[-1]
    print(f"Model {i}: Final GE = {final_ge}:.2f, NTGE = {ntge} ")   
    
    # our goal is the lowest NTGE if 2 models has same NTGE we prefer
    # the one with lower final GE

    if ntge < best_ntge:
        best_ntge = ntge
        best_final_ge = final_ge
        best_model_index = i
    elif ntge == best_ntge and final_ge  < best_final_ge:
        # case: if both NTGE are inf
        best_final_ge = final_ge
        best_model_index = i
        
print("\n--- Decision ---")
if best_model_index == -1:
    print("No results found.")
else:
    print(f"The best performing model is : Model {best_model_index}")
    print(f"   -Its NTGE was: {best_ntge}")
    print(f"   -Its Final GE was: {best_final_ge:.2f}")
    
    best_config = np.load(os.path.join(results_dir, f"model_configuration_{best_model_index}.npy"), allow_pickle=True).item()
    print("\n   -Its winning configuration was: ")
    for key, value in best_config.items():
        print(f"   -{key}: {value}")

