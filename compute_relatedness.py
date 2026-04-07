import torch
from torch.utils.data import DataLoader

from cnn_autoencoder_layer_relusig import ExpertGateAutoencoder

def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def load_autoencoder_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = ExpertGateAutoencoder(
        input_size=ckpt["input_size"],
        code_size=ckpt["code_size"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

def compute_relatedness(first_model, second_model, dataloader, device="cpu"):
    first_model.eval()
    second_model.eval()
    first_model.to(device)
    second_model.to(device)

    first_errors = []
    second_errors = []
    acc_t2_count = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x = _unpack_batch(batch).to(device).float()

            err1 = first_model.reconstruction_error(x, reduction="none")
            err2 = second_model.reconstruction_error(x, reduction="none")

            first_errors.extend(err1.cpu().tolist())
            second_errors.extend(err2.cpu().tolist())

            acc_t2_count += (err2 < err1).sum().item()
            total += x.size(0)

    first_avg_err = sum(first_errors) / len(first_errors)
    second_avg_err = sum(second_errors) / len(second_errors)
    acc_t2 = 100.0 * acc_t2_count / max(total, 1)

    confusion = (first_avg_err - second_avg_err) * 100.0 / max(second_avg_err, 1e-12)
    relatedness = 100.0 - confusion

    return {
        "acc_t2": acc_t2,
        "first_avg_err": first_avg_err,
        "second_avg_err": second_avg_err,
        "confusion": confusion,
        "relatedness": relatedness,
    }

if __name__ == "__main__":
    # Example usage:
    # python compute_relatedness.py
    import numpy as np
    from cnn_autoencoder_layer_relusig import FeatureDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_size = 100
    x_old = np.random.randn(200, input_size).astype("float32")
    x_new = (np.random.randn(200, input_size) + 0.5).astype("float32")

    ds_old = FeatureDataset(x_old)
    ds_new = FeatureDataset(x_new)

    loader_new = DataLoader(ds_new, batch_size=32, shuffle=False)

    m1 = ExpertGateAutoencoder(input_size=input_size, code_size=20).to(device)
    m2 = ExpertGateAutoencoder(input_size=input_size, code_size=20).to(device)

    result = compute_relatedness(m1, m2, loader_new, device=device)
    print(result)
