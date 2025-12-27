import pandas as pd
import numpy as np
import mindspore as ms
from mindspore import nn, context, save_checkpoint, load_checkpoint, load_param_into_net, Tensor, Parameter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import random

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
RESULTS_FILE = "./experiment1/experiment_results_final.txt"

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)

SEED = 42
setup_seed(SEED)

class FraudNet(nn.Cell):
    def __init__(self):
        super(FraudNet, self).__init__()
        # BatchNorm helps align the features from different banks
        self.bn1 = nn.BatchNorm1d(6)
        self.dense1 = nn.Dense(6, 64)
        self.relu = nn.ReLU()
        
        self.bn2 = nn.BatchNorm1d(64)
        self.dense2 = nn.Dense(64, 32)
        
        self.dense3 = nn.Dense(32, 2)

    def construct(self, x):
        x = self.bn1(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return x

# --- DATA GENERATION ---
def generate_dataset(filename, n_samples, fraud_type):
    """
    Generate fraud detection datasets where individual models are ACTIVELY MISLED.
    
    New strategy:
    - Bank A learns: "High amount = Fraud" (regardless of balance)
    - Bank B learns: "High balance = Fraud" (regardless of amount)  
    - Global frauds: NEITHER high amount NOR high balance (moderate values)
    - Global legit: BOTH high amount AND high balance
    
    Result: Individual models will ACTIVELY misclassify global data
    """
    print(f"Generating {filename} ({fraud_type})...")
    data = []
    
    for _ in range(n_samples):
        is_fraud = np.random.random() < 0.3
        
        if fraud_type == "HIGH_VALUE":
            # BANK A: High amount = Fraud (balance varies)
            if is_fraud:
                row = {
                    "type": np.random.choice(["TRANSFER", "CASH_OUT"]),
                    "amount": np.random.uniform(15000, 25000),      # VERY HIGH
                    "oldbalanceOrg": np.random.uniform(1000, 10000), # Varies
                    "newbalanceOrig": 0,
                    "oldbalanceDest": np.random.uniform(100, 1000),
                    "newbalanceDest": 0,
                    "isFraud": 1
                }
            else:
                # Non-fraud: Low-to-medium amount (balance varies)
                row = {
                    "type": np.random.choice(["PAYMENT", "TRANSFER"]),
                    "amount": np.random.uniform(100, 5000),         # LOW-MED
                    "oldbalanceOrg": np.random.uniform(1000, 20000), # Varies widely
                    "newbalanceOrig": 0,
                    "oldbalanceDest": np.random.uniform(100, 1000),
                    "newbalanceDest": 0,
                    "isFraud": 0
                }
                
        elif fraud_type == "HIGH_FREQ":
            # BANK B: High balance = Fraud (amount varies)
            if is_fraud:
                row = {
                    "type": np.random.choice(["TRANSFER", "CASH_OUT"]),
                    "amount": np.random.uniform(1000, 10000),       # Varies
                    "oldbalanceOrg": np.random.uniform(15000, 25000), # VERY HIGH
                    "newbalanceOrig": 0,
                    "oldbalanceDest": np.random.uniform(100, 1000),
                    "newbalanceDest": 0,
                    "isFraud": 1
                }
            else:
                # Non-fraud: Low-to-medium balance (amount varies)
                row = {
                    "type": np.random.choice(["PAYMENT", "CASH_OUT"]),
                    "amount": np.random.uniform(100, 20000),        # Varies widely
                    "oldbalanceOrg": np.random.uniform(500, 5000),  # LOW-MED
                    "newbalanceOrig": 0,
                    "oldbalanceDest": np.random.uniform(100, 1000),
                    "newbalanceDest": 0,
                    "isFraud": 0
                }

        elif fraud_type == "MIXED":
            # GLOBAL: Flip the pattern!
            if is_fraud:
                # Fraud: MEDIUM amount + MEDIUM balance
                # Bank A thinks: "amount not high enough → legit"
                # Bank B thinks: "balance not high enough → legit"
                row = {
                    "type": np.random.choice(["TRANSFER", "CASH_OUT"]),
                    "amount": np.random.uniform(5000, 10000),       # MEDIUM
                    "oldbalanceOrg": np.random.uniform(5000, 10000), # MEDIUM
                    "newbalanceOrig": 0,
                    "oldbalanceDest": np.random.uniform(100, 1000),
                    "newbalanceDest": 0,
                    "isFraud": 1
                }
            else:
                # Non-fraud: VERY HIGH amount + VERY HIGH balance
                # Bank A thinks: "high amount → fraud!" (WRONG)
                # Bank B thinks: "high balance → fraud!" (WRONG)
                # Both models will FALSE POSITIVE on these
                row = {
                    "type": "PAYMENT",
                    "amount": np.random.uniform(15000, 25000),      # VERY HIGH
                    "oldbalanceOrg": np.random.uniform(15000, 25000), # VERY HIGH
                    "newbalanceOrig": 0,
                    "oldbalanceDest": np.random.uniform(100, 1000),
                    "newbalanceDest": 0,
                    "isFraud": 0
                }
                    
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

# --- HELPER: LOCAL STANDARDIZATION ---
def preprocess_data(df, scaler=None):
    # 1. Map Type
    df['type'] = df['type'].apply(lambda x: 1 if x in ['TRANSFER', 'CASH_OUT'] else 0)
    
    # 2. Extract Features
    cols = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    X = df[cols].values.astype(np.float32)
    y = df['isFraud'].values.astype(np.int32)
    
    # 3. Z-SCORE NORMALIZATION
    # Each bank calculates its OWN mean/std. This aligns the distributions.
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        # For testing, we ideally use the training scaler, 
        # but in this PoC, fitting on test data simulates the "local view" of the test set.
        X = scaler.fit_transform(X) 
        
    return X, y, scaler

# --- TRAINING FUNCTION ---
def train_model(data_file, output_ckpt, epochs=20):
    print(f"Training on {data_file} -> {output_ckpt}...")
    df = pd.read_csv(data_file)
    X, y, _ = preprocess_data(df) # Each bank uses its own scaler during training
    
    net = FraudNet()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    train_net = nn.TrainOneStepCell(nn.WithLossCell(net, loss_fn), optimizer)
    train_net.set_train()
    
    batch_size = 32
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            end = i + batch_size
            train_net(Tensor(X[i:end]), Tensor(y[i:end]))
            
    save_checkpoint(net, output_ckpt)

# --- 4. AGGREGATION FUNCTION ---
def aggregate_models(ckpt_a, ckpt_b, output_ckpt):
    print(f"Aggregating models -> {output_ckpt}...")
    params_a = load_checkpoint(ckpt_a)
    params_b = load_checkpoint(ckpt_b)
    new_params = []
    
    for key in params_a:
        wa = params_a[key].asnumpy()
        wb = params_b[key].asnumpy()
        w_avg = (wa + wb) / 2.0
        new_params.append({"name": key, "data": Parameter(Tensor(w_avg), name=key)})
        
    save_checkpoint(new_params, output_ckpt)

# --- 5. EVALUATION HELPER ---
def get_accuracy(ckpt_path, csv_file):
    df = pd.read_csv(csv_file)
    X, y, _ = preprocess_data(df) # Treat test data as its own "local" environment
    
    net = FraudNet()
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    
    output = net(Tensor(X))
    predicted = np.argmax(output.asnumpy(), axis=1)
    
    return accuracy_score(y, predicted) * 100

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Generate
    generate_dataset("./experiment1/bank_a_train.csv", 1000, "HIGH_VALUE")
    generate_dataset("./experiment1/bank_b_train.csv", 1000, "HIGH_FREQ")
    generate_dataset("./experiment1/global_test.csv",  500,  "MIXED")
    
    # Train
    train_model("./experiment1/bank_a_train.csv", "./experiment1/model_bank_a.ckpt")
    train_model("./experiment1/bank_b_train.csv", "./experiment1/model_bank_b.ckpt")
    
    # Aggregate
    aggregate_models("./experiment1/model_bank_a.ckpt", "./experiment1/model_bank_b.ckpt", "./experiment1/model_global.ckpt")
    
    # Evaluate
    print("\n--- Running Evaluation Matrix ---")
    results = []
    
    # 1. Bank A
    acc_a_on_a = get_accuracy("./experiment1/model_bank_a.ckpt", "./experiment1/bank_a_train.csv")
    acc_a_on_b = get_accuracy("./experiment1/model_bank_a.ckpt", "./experiment1/bank_b_train.csv")
    acc_a_on_g = get_accuracy("./experiment1/model_bank_a.ckpt", "./experiment1/global_test.csv")
    results.append(f"Bank A Model on Bank A Data:     {acc_a_on_a:.2f}%")
    results.append(f"Bank A Model on Bank B Data:     {acc_a_on_b:.2f}%")
    results.append(f"Bank A Model on Global Dataset:  {acc_a_on_g:.2f}%")
    results.append("-" * 40)
    
    # 2. Bank B
    acc_b_on_a = get_accuracy("./experiment1/model_bank_b.ckpt", "./experiment1/bank_a_train.csv")
    acc_b_on_b = get_accuracy("./experiment1/model_bank_b.ckpt", "./experiment1/bank_b_train.csv")
    acc_b_on_g = get_accuracy("./experiment1/model_bank_b.ckpt", "./experiment1/global_test.csv")
    results.append(f"Bank B Model on Bank A Data:     {acc_b_on_a:.2f}%")
    results.append(f"Bank B Model on Bank B Data:     {acc_b_on_b:.2f}%")
    results.append(f"Bank B Model on Global Dataset:  {acc_b_on_g:.2f}%")
    results.append("-" * 40)
    
    # 3. Global
    acc_g_on_a = get_accuracy("./experiment1/model_global.ckpt", "./experiment1/bank_a_train.csv")
    acc_g_on_b = get_accuracy("./experiment1/model_global.ckpt", "./experiment1/bank_b_train.csv")
    acc_g_on_g = get_accuracy("./experiment1/model_global.ckpt", "./experiment1/global_test.csv")
    results.append(f"Global Model on Bank A Data:     {acc_g_on_a:.2f}%")
    results.append(f"Global Model on Bank B Data:     {acc_g_on_b:.2f}%")
    results.append(f"Global Model on Global Dataset:  {acc_g_on_g:.2f}%")
    
    with open("./experiment1/experiment_results_final.txt", "w") as f:
        for line in results:
            print(line)       
            f.write(line + "\n") 
            
    print(f"\nResults saved to experiment_results_final.txt")