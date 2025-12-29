import pandas as pd
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model, load_checkpoint, load_param_into_net
from adapter import DataHarmonizer
from model import FraudNet

# 1. Setup Environment
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class IterableData:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    def __len__(self):
        return len(self.X)

def run_evaluation_matrix():
    print("--- 📊 STARTING COMPREHENSIVE EVALUATION MATRIX ---")
    
    # 2. Define the Datasets (The Scenarios)
    # Each dataset needs its specific Adapter Config to be readable
    datasets = [
        {
            "name": "Bank A Data (Clean)", 
            "path": "experiment2/bank_a_raw.csv", 
            "map": {}, 
            "scale": 1.0
        },
        {
            "name": "Bank B Data (Renamed)", 
            "path": "experiment2/bank_b_raw.csv", 
            "map": {'txn_val': 'amount', 'old_bal': 'oldbalanceOrg'}, 
            "scale": 1.0
        },
        {
            "name": "Bank C Data (Cents)", 
            "path": "experiment2/bank_c_raw.csv", 
            "map": {'amount_cents': 'amount'}, 
            "scale": 0.01
        }
    ]

    # 3. Define the Models to Test
    models_to_test = [
        {"name": "Bank A Model", "ckpt": "experiment2/bank_a.ckpt"},
        {"name": "Bank B Model", "ckpt": "experiment2/bank_b.ckpt"},
        {"name": "Bank C Model", "ckpt": "experiment2/bank_c.ckpt"},
        {"name": "GLOBAL MODEL", "ckpt": "experiment2/global_model.ckpt"}
    ]

    results_buffer = []
    
    # 4. The Loop: For Every Model...
    for model_info in models_to_test:
        print(f"\n🔹 Loading {model_info['name']}...")
        
        # Load Weights
        net = FraudNet()
        try:
            param_dict = load_checkpoint(model_info['ckpt'])
            load_param_into_net(net, param_dict)
        except Exception as e:
            print(f"   ❌ Could not load {model_info['ckpt']}. Skipping.")
            continue

        # Prepare Metric Calculators
        # Note: 'binary' handles the 2-class setup automatically
        metrics = {
            'acc': nn.Accuracy(),
            'prec': nn.Precision(), 
            'recall': nn.Recall()
        }
        
        model = Model(net, loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True), metrics=metrics)

        # 5. ...Test on Every Dataset
        for data_info in datasets:
            print(f"   -> Testing on {data_info['name']}...")
            
            # A. Adapter Step (Harmonize Data)
            raw_df = pd.read_csv(data_info['path'])
            adapter = DataHarmonizer(schema_map=data_info['map'], scaling_factor=data_info['scale'])
            X, y = adapter.process(raw_df)
            
            # B. Create Dataset
            dataset_generator = IterableData(X, y)
            ms_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=["data", "label"])
            ms_dataset = ms_dataset.batch(batch_size=64)
            
            # C. Evaluate
            res = model.eval(ms_dataset)
            
            # D. Format Result
            # Accuracy is usually float, Precision/Recall might be lists or arrays depending on version
            acc = round(res['acc'], 4)
            prec = round(res['prec'][1], 4) if isinstance(res['prec'], (list, np.ndarray)) else round(res['prec'], 4)
            rec = round(res['recall'][1], 4) if isinstance(res['recall'], (list, np.ndarray)) else round(res['recall'], 4)

            log_line = (f"| {model_info['name']:<15} | {data_info['name']:<20} "
                        f"| Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} |")
            
            print(log_line)
            results_buffer.append(log_line)

    # 6. Save to File
    with open("experiment2/evaluation_matrix.txt", "w") as f:
        f.write("--- FEDERATED LEARNING EVALUATION MATRIX ---\n")
        f.write("| Model Name      | Dataset Name         | Accuracy  | Precision | Recall |\n")
        f.write("|-----------------|----------------------|-----------|-----------|--------|\n")
        for line in results_buffer:
            f.write(line + "\n")
            
    print("\n✅ Evaluation Matrix saved to 'experiment2/evaluation_matrix.txt'")

if __name__ == "__main__":
    run_evaluation_matrix()