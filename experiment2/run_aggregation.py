import mindspore
import numpy as np
from mindspore import save_checkpoint, load_checkpoint, Parameter, Tensor
from model import FraudNet

def robust_aggregation():
    print("\n--- 🛡️ ROBUST FEDERATION SERVER (FedMedian) ---")
    
    global_net = FraudNet()
    clients = ['experiment2/bank_a.ckpt', 'experiment2/bank_b.ckpt', 'experiment2/bank_c.ckpt']
    updates = []

    # 1. Load all client parameters
    for ckpt_path in clients:
        print(f"[Server] Loading update from {ckpt_path}...")
        updates.append(load_checkpoint(ckpt_path))
    
    print("[Server] Computing FedMedian (Robust to outliers)...")
    new_params = {}
    
    # 2. Iterate through every layer (weight/bias)
    for param_name in updates[0].keys():
        # Collect this layer's weights from ALL banks into a list
        # Shape: [3, 64, 7] (3 banks, 64 neurons, 7 inputs)
        layers_stacked = []
        for update in updates:
            layers_stacked.append(update[param_name].data.asnumpy())
            
        layers_stacked = np.array(layers_stacked)
        
        # 3. Calculate Median along the 'Bank' axis (axis 0)
        # This finds the median value for EVERY single specific weight
        median_weight = np.median(layers_stacked, axis=0)
        
        new_params[param_name] = Parameter(Tensor(median_weight), name=param_name)
    
    # 4. Save
    mindspore.load_param_into_net(global_net, new_params)
    save_checkpoint(global_net, "experiment2/global_model.ckpt")
    print("✅ GLOBAL MODEL UPDATED (Robust).")

if __name__ == "__main__":
    robust_aggregation()