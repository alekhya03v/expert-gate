import os
import torch
import torch.nn.functional as F

def _unpack_x(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def cnn_train_adagrad_oneLayer(net, train_loader, val_loader, **kwargs):
    opts = {
        "numEpochs": 100,
        "batchSize": 12,
        "useGpu": torch.cuda.is_available(),
        "learningRate": 1e-2,
        "continue_train": False,
        "expDir": "./checkpoints/default",
        "weightDecay": 5e-4,
        "delta": 1e-8,
        "display": 1,
        "snapshot": 1,
        "test_interval": 1,
    }
    opts.update(kwargs)

    os.makedirs(opts["expDir"], exist_ok=True)
    device = "cuda" if opts["useGpu"] and torch.cuda.is_available() else "cpu"
    net = net.to(device)

    optimizer = torch.optim.Adagrad(
        net.parameters(),
        lr=opts["learningRate"],
        weight_decay=opts["weightDecay"],
        eps=opts["delta"]
    )

    info = {"train": {"objective": []}, "val": {"objective": []}}
    start_epoch = 1

    if opts["continue_train"]:
        ckpts = [f for f in os.listdir(opts["expDir"]) if f.startswith("net-epoch-") and f.endswith(".pt")]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
            last = os.path.join(opts["expDir"], ckpts[-1])
            saved = torch.load(last, map_location=device)
            net.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            info = saved["info"]
            start_epoch = saved["epoch"] + 1
            print(f"Resuming from epoch {saved['epoch']}")

    for epoch in range(start_epoch, opts["numEpochs"] + 1):
        net.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            x = _unpack_x(batch).to(device).float()
            x_hat = net(x)
            loss = F.binary_cross_entropy(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_count += x.size(0)

            if opts["display"]:
                print(f"training: epoch {epoch:03d}, batch {batch_idx:03d}, loss {loss.item():.6f}")

        train_loss = train_loss_sum / max(train_count, 1)
        info["train"]["objective"].append(train_loss)

        if epoch == 1 or epoch % opts["test_interval"] == 0 or epoch == opts["numEpochs"]:
            net.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = _unpack_x(batch).to(device).float()
                    x_hat = net(x)
                    loss = F.binary_cross_entropy(x_hat, x)
                    val_loss_sum += loss.item() * x.size(0)
                    val_count += x.size(0)

            val_loss = val_loss_sum / max(val_count, 1)
            info["val"]["objective"].append(val_loss)
            print(f"Epoch {epoch:03d}/{opts['numEpochs']} | train={train_loss:.6f} | val={val_loss:.6f}")

        if epoch % opts["snapshot"] == 0 or epoch == opts["numEpochs"]:
            save_path = os.path.join(opts["expDir"], f"net-epoch-{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "info": info,
                "input_size": getattr(net, "input_size", None),
                "code_size": getattr(net, "code_size", None),
            }, save_path)

    return net, info
