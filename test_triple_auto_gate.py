import torch
from torch.utils.data import DataLoader

from compute_relatedness import load_autoencoder_checkpoint

def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def test_triple_auto_gate(encoders, dataloaders_by_task, device="cpu"):
    """
    encoders: list of trained autoencoder models
    dataloaders_by_task: one test loader per task/domain
    """
    for model in encoders:
        model.to(device)
        model.eval()

    task_gate_accuracies = []

    with torch.no_grad():
        for true_task_id, loader in enumerate(dataloaders_by_task):
            correct = 0
            total = 0

            for batch in loader:
                x = _unpack_batch(batch).to(device).float()

                errs = []
                for model in encoders:
                    e = model.reconstruction_error(x, reduction="none")
                    errs.append(e.unsqueeze(1))

                errs = torch.cat(errs, dim=1)          # [B, num_tasks]
                pred_task = torch.argmin(errs, dim=1)  # minimum reconstruction error

                correct += (pred_task == true_task_id).sum().item()
                total += x.size(0)

            acc = 100.0 * correct / max(total, 1)
            task_gate_accuracies.append(acc)
            print(f"Task {true_task_id}: gate accuracy = {acc:.2f}%")

    return task_gate_accuracies

if __name__ == "__main__":
    import numpy as np
    from cnn_autoencoder_layer_relusig import ExpertGateAutoencoder, FeatureDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 64

    x1 = np.random.randn(150, input_size).astype("float32")
    x2 = (np.random.randn(150, input_size) + 1.0).astype("float32")
    x3 = (np.random.randn(150, input_size) - 1.0).astype("float32")

    ds1 = FeatureDataset(x1)
    ds2 = FeatureDataset(x2)
    ds3 = FeatureDataset(x3)

    l1 = DataLoader(ds1, batch_size=16, shuffle=False)
    l2 = DataLoader(ds2, batch_size=16, shuffle=False)
    l3 = DataLoader(ds3, batch_size=16, shuffle=False)

    ae1 = ExpertGateAutoencoder(input_size, 20).to(device)
    ae2 = ExpertGateAutoencoder(input_size, 20).to(device)
    ae3 = ExpertGateAutoencoder(input_size, 20).to(device)

    accs = test_triple_auto_gate([ae1, ae2, ae3], [l1, l2, l3], device=device)
    print(accs)
