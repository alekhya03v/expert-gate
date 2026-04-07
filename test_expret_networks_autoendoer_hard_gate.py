import torch
import torch.nn as nn

class SimpleExpert(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def test_expret_networks_autoendoer_hard_gate(encoders, experts, dataloaders_by_task, device="cpu"):
    for enc in encoders:
        enc.to(device).eval()
    for exp in experts:
        exp.to(device).eval()

    results = []

    with torch.no_grad():
        for task_id, loader in enumerate(dataloaders_by_task):
            gate_correct = 0
            cls_correct = 0
            total = 0

            for batch in loader:
                x, y = batch
                x = x.to(device).float()
                y = y.to(device)

                errs = []
                for enc in encoders:
                    err = enc.reconstruction_error(x, reduction="none")
                    errs.append(err.unsqueeze(1))
                errs = torch.cat(errs, dim=1)

                chosen_task = torch.argmin(errs, dim=1)
                gate_correct += (chosen_task == task_id).sum().item()

                for i in range(x.size(0)):
                    expert_id = int(chosen_task[i].item())
                    logits = experts[expert_id](x[i:i+1])
                    pred = logits.argmax(dim=1)
                    cls_correct += (pred == y[i:i+1]).sum().item()
                    total += 1

            gate_acc = 100.0 * gate_correct / max(total, 1)
            cls_acc = 100.0 * cls_correct / max(total, 1)

            results.append({
                "task_id": task_id + 1,
                "gate_accuracy": gate_acc,
                "classification_accuracy": cls_acc
            })

            print(f"Task {task_id + 1}: gate={gate_acc:.2f}% | cls={cls_acc:.2f}%")

    return results
