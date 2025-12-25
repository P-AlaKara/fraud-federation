import mindspore as ms
from mindspore import nn, save_checkpoint

# Define the Architecture (Must match the Bank's model exactly)
class FraudNet(nn.Cell):
    def __init__(self):
        super(FraudNet, self).__init__()
        # Input: 6 features -> Output: 2 classes (Fraud/Not Fraud)
        self.dense1 = nn.Dense(6, 32)
        self.relu = nn.ReLU()
        self.dense2 = nn.Dense(32, 16)
        self.dense3 = nn.Dense(16, 2)

    def construct(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    print("Initializing Global Model (Genesis)...")
    net = FraudNet()
    
    # Save the initial random weights
    save_checkpoint(net, "./models/global_model.ckpt")
    print("Success: ./models/global_model.ckpt created.")