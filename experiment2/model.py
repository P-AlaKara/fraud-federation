import mindspore.nn as nn
from mindspore.common.initializer import Normal

class FraudNet(nn.Cell):
    """
    Standardized AI Model for all Banks.
    Input Layer: 7 Neurons (Standardized Features)
    Output Layer: 2 Neurons (Legit vs Fraud)
    """
    def __init__(self):
        super(FraudNet, self).__init__()
        # Architecture: 7 Input -> 64 Hidden -> 32 Hidden -> 2 Output
        self.fc1 = nn.Dense(7, 64, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(64, 32, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(32, 2, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x