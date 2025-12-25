import os
import mindspore as ms
from mindspore import save_checkpoint, load_checkpoint, Parameter, Tensor
import numpy as np

# Configuration
UPLOAD_FOLDER = './uploads'
GLOBAL_MODEL_PATH = './models/global_model.ckpt'

def perform_aggregation():
    """
    Loads all .ckpt files from /uploads, averages their weights,
    updates the global model, and cleans up the uploads.
    """
    print("--- Starting Aggregation ---")
    
    # 1. Identify all uploaded checkpoints
    ckpt_files = [
        os.path.join(UPLOAD_FOLDER, f) 
        for f in os.listdir(UPLOAD_FOLDER) 
        if f.endswith('.ckpt')
    ]
    
    if not ckpt_files:
        print("No files to aggregate.")
        return

    num_participants = len(ckpt_files)
    print(f"Aggregating updates from {num_participants} banks...")

    # 2. Load all parameters into a dictionary
    # Structure: { "conv1.weight": [Tensor_BankA, Tensor_BankB], ... }
    param_accumulator = {}
    
    # Initialize with the schema of the first file
    first_params = load_checkpoint(ckpt_files[0])
    for key in first_params:
        param_accumulator[key] = []

    # Load and accumulate
    for ckpt in ckpt_files:
        params = load_checkpoint(ckpt)
        for key, parameter in params.items():
            # Convert Parameter to Numpy for easy math, then store
            param_accumulator[key].append(parameter.asnumpy())

    # 3. Perform Federated Averaging (FedAvg)
    # Formula: W_global = Sum(W_local) / N
    new_global_params = []
    
    for key, weights_list in param_accumulator.items():
        # Stack them and calculate mean
        stacked_weights = np.array(weights_list)
        averaged_weight = np.mean(stacked_weights, axis=0)
        
        # Convert back to MindSpore Parameter
        ms_param = Parameter(Tensor(averaged_weight), name=key)
        new_global_params.append({"name": key, "data": ms_param})

    # 4. Save the new Global Model
    save_checkpoint(new_global_params, GLOBAL_MODEL_PATH)
    print(f"Global Model updated and saved to {GLOBAL_MODEL_PATH}")

    # 5. Cleanup: Delete processed uploads
    for f in ckpt_files:
        os.remove(f)
    print("Uploads cleaned. Ready for next round.")

    return True