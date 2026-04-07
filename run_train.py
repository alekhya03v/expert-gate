import numpy as np
import torch

from cnn_autoencoder_layer_relusig import cnn_autoencoder_layer_relusig, FeatureDataset
from cnn_train_adagrad_oneLayer import cnn_train_adagrad_oneLayer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Dummy feature data
    # Replace this with your real extracted features
    num_samples = 500
    input_size = 100
    x = np.random.randn(num_samples, input_size).astype("float32")

    dataset = FeatureDataset(x)

    net, opts = cnn_autoencoder_layer_relusig(
        input_size=input_size,
        code_size=20,
        expDir="./checkpoints/task1_autoencoder",
        useGpu=torch.cuda.is_available()
    )

    net, info = cnn_train_adagrad_oneLayer(
        net=net,
        input_net=None,
        imdb=dataset,
        batchSize=32,
        numEpochs=10,
        learningRate=1e-2,
        expDir="./checkpoints/task1_autoencoder",
        useGpu=torch.cuda.is_available(),
        test_interval=1,
        snapshot=1
    )

    print("Training complete.")
    print("Train losses:", info["train"]["objective"])
    print("Val losses:", info["val"]["objective"])

if __name__ == "__main__":
    main()
