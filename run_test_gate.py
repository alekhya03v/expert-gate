import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_autoencoder_layer_relusig import ExpertGateAutoencoder, FeatureDataset
from test_triple_auto_gate import test_triple_auto_gate

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    input_size = 64

    # Dummy task-wise features
    x1 = np.random.randn(200, input_size).astype("float32")
    x2 = (np.random.randn(200, input_size) + 1.0).astype("float32")
    x3 = (np.random.randn(200, input_size) - 1.0).astype("float32")

    ds1 = FeatureDataset(x1)
    ds2 = FeatureDataset(x2)
    ds3 = FeatureDataset(x3)

    l1 = DataLoader(ds1, batch_size=32, shuffle=False)
    l2 = DataLoader(ds2, batch_size=32, shuffle=False)
    l3 = DataLoader(ds3, batch_size=32, shuffle=False)

    ae1 = ExpertGateAutoencoder(input_size=input_size, code_size=20).to(device)
    ae2 = ExpertGateAutoencoder(input_size=input_size, code_size=20).to(device)
    ae3 = ExpertGateAutoencoder(input_size=input_size, code_size=20).to(device)

    accs = test_triple_auto_gate(
        encoders=[ae1, ae2, ae3],
        dataloaders_by_task=[l1, l2, l3],
        device=device
    )

    print("Gate accuracies:", accs)

if __name__ == "__main__":
    main()
