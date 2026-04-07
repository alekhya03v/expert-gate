import torch
from torch.utils.data import DataLoader

from cnn_autoencoder_layer_relusig import ExpertGateAutoencoder
from dataset_utils import build_dataset, split_indices_from_sets

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
    first_model.eval().to(device)
    second_model.eval().to(device)

    all_first_err = []
    all_second_err = []
    acc_t2 = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device).float() if isinstance(batch, (list, tuple)) else batch.to(device).float()
            first_err = first_model.reconstruction_error(x, reduction="none")
            second_err = second_model.reconstruction_error(x, reduction="none")

            all_first_err.extend(first_err.cpu().tolist())
            all_second_err.extend(second_err.cpu().tolist())

            acc_t2 += (second_err < first_err).sum().item()
            total += x.size(0)

    acc_t2 = 100.0 * acc_t2 / max(total, 1)
    first_avg_err = sum(all_first_err) / len(all_first_err)
    second_avg_err = sum(all_second_err) / len(all_second_err)
    confusion = (first_avg_err - second_avg_err) * 100.0 / max(second_avg_err, 1e-12)
    relatedness = 100.0 - confusion

    return {
        "acc_t2": acc_t2,
        "first_avg_err": first_avg_err,
        "second_avg_err": second_avg_err,
        "confusion": confusion,
        "relatedness": relatedness,
    }
