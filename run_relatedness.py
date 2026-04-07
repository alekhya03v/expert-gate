import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_autoencoder_layer_relusig import ExpertGateAutoencoder, FeatureDataset
from compute_relatedness import compute_relatedness

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    input_size = 100

    # Dummy data
    x_new_task = (np.random.randn(300, input_size) + 0.5).astype("float32")
    dataset = FeatureDataset(x_new_task)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Two example autoencoders
    model1 = ExpertGateAutoencoder(input_size=input_size, code_size=20).to(device)
    model2 = ExpertGateAutoencoder(input_size=input_size, code_size=20).to(device)

    result = compute_relatedness(model1, model2, loader, device=device)

    print("Relatedness result:")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
