import mindspore
from mindspore import save_checkpoint
from model import FraudNet

def init_genesis_model():
    print("--- CREATING GENESIS MODEL ---")
    net = FraudNet()
    
    # Save this initial random state
    save_checkpoint(net, "experiment2/global_model.ckpt")
    print("global_model.ckpt created. All banks must start from this file!")

if __name__ == "__main__":
    init_genesis_model()