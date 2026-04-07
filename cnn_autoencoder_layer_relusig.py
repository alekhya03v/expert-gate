import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def sigmoid_tensor(x):
    return torch.sigmoid(x)

def sparse_linear_init(layer: nn.Linear, non_zero_per_output: int = 15):
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.zero_()

        out_features, in_features = layer.weight.shape
        k = min(non_zero_per_output, in_features)

        for out_idx in range(out_features):
            idx = torch.randperm(in_features)[:k]
            layer.weight[out_idx, idx] = torch.randn(k) * 0.01

class ExpertGateAutoencoder(nn.Module):
    """
    One-layer autoencoder:
    input -> code -> sigmoid(reconstruction)

    MATLAB repo uses:
    - sparse initialization
    - code_size = latent dimension
    - sigmoid decoder output
    """
    def __init__(self, input_size: int, code_size: int = 500, sparse_k: int = 15):
        super().__init__()
        self.input_size = input_size
        self.code_size = code_size

        self.encoder = nn.Linear(input_size, code_size)
        self.decoder = nn.Linear(code_size, input_size)

        sparse_linear_init(self.encoder, sparse_k)
        sparse_linear_init(self.decoder, sparse_k)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = torch.sigmoid(self.decoder(z))
        return x_hat

    def reconstruction_error(self, x, reduction="none"):
        x_hat = self.forward(x)
        err = ((x_hat - x) ** 2).sum(dim=1)
        if reduction == "mean":
            return err.mean()
        return err

class FeatureDataset(Dataset):
    """
    Dataset for pre-extracted features.

    If you have raw features per sample:
        features: [N, D]
    Optionally:
        labels: [N]
        mean/std: normalization stats
    """
    def __init__(self, features, labels=None, mean=None, std=None, apply_sigmoid=True):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)

        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        self.std = None if std is None else torch.as_tensor(std, dtype=torch.float32)

        self.apply_sigmoid = apply_sigmoid

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        if self.apply_sigmoid:
            x = torch.sigmoid(x)

        if self.labels is None:
            return x
        return x, self.labels[idx]

def get_default_autoencoder_opts():
    return {
        "useValidation": True,
        "imdbPath": "./data/scenes/encoder_input_scenes_imdb.pt",
        "expDir": "./Scenes/autoencoder/onelayer_direct_input_encodernorm/",
        "code_size": 500,
        "input_size": 43264,
        "batchSize": 12,
        "initial_encoder": None,
        "errorType": "euclideanloss",
        "display": 1,
        "delta": 1e-8,
        "continue_train": False,
        "learningRate": 1e-2,
        "numEpochs": 100,
        "plotDiagnostics": False,
        "prefetch": False,
        "snapshot": 1,
        "sync": True,
        "test_interval": 1,
        "useGpu": True,
        "weightDecay": 5e-4,
    }

def cnn_autoencoder_layer_relusig(**kwargs):
    """
    Python replacement for cnn_autoencoder_layer_relusig.m

    Builds a one-layer autoencoder and returns:
        net, opts
    """
    opts = get_default_autoencoder_opts()
    opts.update(kwargs)

    if not os.path.exists(opts["expDir"]):
        os.makedirs(opts["expDir"], exist_ok=True)

    if opts["initial_encoder"] is not None and os.path.exists(opts["initial_encoder"]):
        checkpoint = torch.load(opts["initial_encoder"], map_location="cpu")
        net = ExpertGateAutoencoder(
            input_size=checkpoint["input_size"],
            code_size=checkpoint["code_size"]
        )
        net.load_state_dict(checkpoint["model_state"])
    else:
        net = ExpertGateAutoencoder(
            input_size=opts["input_size"],
            code_size=opts["code_size"]
        )

    return net, opts
