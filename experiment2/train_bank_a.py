import mindspore
import pandas as pd
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model, save_checkpoint
from adapter import DataHarmonizer
from model import FraudNet

# Context Setup
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Dataset Generator
class IterableData:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    def __len__(self):
        return len(self.X)

def run_bank_a():
    print("\n--- 🏦 BANK A (M-Pesa) STARTED ---")
    
    # 1. Load Data
    raw_df = pd.read_csv('experiment2/bank_a_raw.csv')
    
    # 2. Adapter (Standard Config)
    adapter = DataHarmonizer() 
    X, y = adapter.process(raw_df)
    
    # 3. Create MindSpore Dataset
    dataset_generator = IterableData(X, y)
    ms_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=["data", "label"])
    ms_dataset = ms_dataset.batch(batch_size=64)
    
    # 4. Train
    net = FraudNet()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Adam(net.trainable_params(), learning_rate=0.001)

    print("[Federation] Loading global weights (Synced Initialization)...")
    try:
        param_dict = mindspore.load_checkpoint("experiment2/global_model.ckpt")
        mindspore.load_param_into_net(net, param_dict)
    except Exception as e:
        print("⚠️ Warning: Could not load global model. Are you sure you ran init_global.py?")
    
    print(f"[Bank A] Training on {len(X)} rows...")

    model = Model(net, loss, opt, metrics={'acc'})
    model.train(epoch=3, train_dataset=ms_dataset)
    
    # 5. Evaluate & Log
    acc = model.eval(ms_dataset)
    print(f"[Bank A] Local Accuracy: {acc}")
    
    with open("experiment2/results.txt", "a") as f:
        f.write(f"Bank A Local Accuracy: {acc['acc']}\n")
    
    # 6. Save Checkpoint
    save_checkpoint(net, "experiment2/bank_a.ckpt")
    print("✅ Bank A Complete.")

if __name__ == "__main__":
    run_bank_a()