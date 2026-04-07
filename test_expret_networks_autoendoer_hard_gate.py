import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def test_expret_networks_autoendoer_hard_gate(encoders, experts, dataloaders_by_task, device="cpu"):
    """
    Hard-gating evaluation:
    1. Use autoencoders to choose the task/expert with minimum reconstruction error
    2. Run only that chosen expert
    3. Measure gate accuracy and classification accuracy

    encoders: list of trained gate autoencoders
    experts:  list of trained expert classifiers
    dataloaders_by_task: list of test loaders, each yields (x, y)
    """
    for m in encoders:
        m.to(device)
        m.eval()

    for e in experts:
        e.to(device)
        e.eval()

    all_results = []

    with torch.no_grad():
        for true_task_id, loader in enumerate(dataloaders_by_task):
            gate_correct = 0
            cls_correct = 0
            total = 0

            for batch in loader:
                x, y = batch
                x = x.to(device).float()
                y = y.to(device)

                errs = []
                for model in encoders:
                    e = model.reconstruction_error(x, reduction="none")
                    errs.append(e.unsqueeze(1))
                errs = torch.cat(errs, dim=1)            # [B, num_tasks]
                chosen_task = torch.argmin(errs, dim=1)  # hard gate

                gate_correct += (chosen_task == true_task_id).sum().item()

                # Run selected expert sample-by-sample
                for i in range(x.size(0)):
                    expert_id = int(chosen_task[i].item())
                    logits = experts[expert_id](x[i:i+1])
                    pred = logits.argmax(dim=1)
                    cls_correct += (pred == y[i:i+1]).sum().item()
                    total += 1

            gate_acc = 100.0 * gate_correct / max(total, 1)
            cls_acc = 100.0 * cls_correct / max(total, 1)

            result = {
                "task_id": true_task_id,
                "gate_accuracy": gate_acc,
                "classification_accuracy": cls_acc,
            }
            all_results.append(result)

            print(
                f"Task {true_task_id}: "
                f"gate_accuracy={gate_acc:.2f}% | "
                f"classification_accuracy={cls_acc:.2f}%"
            )

    return all_results

# Example expert network
class SimpleExpert(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    import numpy as np
    from cnn_autoencoder_layer_relusig import ExpertGateAutoencoder, FeatureDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 64
    num_classes = 5

    x1 = np.random.randn(120, input_size).astype("float32")
    y1 = np.random.randint(0, num_classes, size=(120,))
    x2 = (np.random.randn(120, input_size) + 1.0).astype("float32")
    y2 = np.random.randint(0, num_classes, size=(120,))
    x3 = (np.random.randn(120, input_size) - 1.0).astype("float32")
    y3 = np.random.randint(0, num_classes, size=(120,))

    ds1 = FeatureDataset(x1, y1)
    ds2 = FeatureDataset(x2, y2)
    ds3 = FeatureDataset(x3, y3)

    l1 = DataLoader(ds1, batch_size=16, shuffle=False)
    l2 = DataLoader(ds2, batch_size=16, shuffle=False)
    l3 = DataLoader(ds3, batch_size=16, shuffle=False)

    ae1 = ExpertGateAutoencoder(input_size, 20).to(device)
    ae2 = ExpertGateAutoencoder(input_size, 20).to(device)
    ae3 = ExpertGateAutoencoder(input_size, 20).to(device)

    exp1 = SimpleExpert(input_size, num_classes).to(device)
    exp2 = SimpleExpert(input_size, num_classes).to(device)
    exp3 = SimpleExpert(input_size, num_classes).to(device)

    results = test_expret_networks_autoendoer_hard_gate(
        [ae1, ae2, ae3],
        [exp1, exp2, exp3],
        [l1, l2, l3],
        device=device
    )
    print(results)
