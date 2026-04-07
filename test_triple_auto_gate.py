import torch

def test_triple_auto_gate(encoders, dataloaders_by_task, device="cpu"):
    for model in encoders:
        model.to(device)
        model.eval()

    acc_all = []

    with torch.no_grad():
        for true_task_id, loader in enumerate(dataloaders_by_task):
            correct = 0
            total = 0

            for batch in loader:
                x = batch[0].to(device).float() if isinstance(batch, (list, tuple)) else batch.to(device).float()

                reconstruction_err = []
                for model in encoders:
                    err = model.reconstruction_error(x, reduction="none")
                    reconstruction_err.append(err.unsqueeze(1))

                reconstruction_err = torch.cat(reconstruction_err, dim=1)
                pred_task = torch.argmin(reconstruction_err, dim=1)

                correct += (pred_task == true_task_id).sum().item()
                total += x.size(0)

            acc = 100.0 * correct / max(total, 1)
            acc_all.append(acc)
            print(f"Task {true_task_id + 1}: gate accuracy = {acc:.2f}%")

    return acc_all
