import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from cnn_autoencoder_layer_relusig import ExpertGateAutoencoder

def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def _save_checkpoint(path, model, optimizer, epoch, info, input_size, code_size):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "info": info,
        "input_size": input_size,
        "code_size": code_size,
    }, path)

def cnn_train_adagrad_oneLayer(
    net,
    input_net,
    imdb,
    getBatch=None,
    **kwargs
):
    """
    Python replacement for cnn_train_adagrad_oneLayer.m

    Expected:
        imdb = torch Dataset OR a dict containing dataset under imdb["dataset"]
    """
    opts = {
        "train": None,
        "val": None,
        "numEpochs": 300,
        "batchSize": 256,
        "useGpu": torch.cuda.is_available(),
        "learningRate": 0.001,
        "continue_train": False,
        "expDir": os.path.join("data", "exp"),
        "weightDecay": 5e-4,
        "errorType": "euclideanloss",
        "delta": 1e-8,
        "display": 1,
        "snapshot": 1,
        "test_interval": 1,
        "useValidation": True,
    }
    opts.update(kwargs)

    os.makedirs(opts["expDir"], exist_ok=True)

    if isinstance(imdb, dict):
        dataset = imdb["dataset"]
    else:
        dataset = imdb

    if opts["useValidation"]:
        val_size = max(1, int(0.1 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds = dataset
        val_ds = dataset

    train_loader = DataLoader(train_ds, batch_size=opts["batchSize"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=opts["batchSize"], shuffle=False)

    device = "cuda" if opts["useGpu"] and torch.cuda.is_available() else "cpu"
    net = net.to(device)

    optimizer = torch.optim.Adagrad(
        net.parameters(),
        lr=opts["learningRate"],
        weight_decay=opts["weightDecay"],
        eps=opts["delta"]
    )

    info = {
        "train": {"objective": [], "error": []},
        "val": {"objective": [], "error": []},
    }

    start_epoch = 1

    if opts["continue_train"]:
        checkpoints = [f for f in os.listdir(opts["expDir"]) if f.startswith("net-epoch-") and f.endswith(".pt")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
            last_ckpt = os.path.join(opts["expDir"], checkpoints[-1])
            ckpt = torch.load(last_ckpt, map_location=device)
            net.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            info = ckpt["info"]
            start_epoch = ckpt["epoch"] + 1
            print(f"Resuming from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, opts["numEpochs"] + 1):
        net.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            x = _unpack_batch(batch).to(device).float()

            x_hat = net(x)
            loss = F.binary_cross_entropy(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_count += x.size(0)

            if opts["display"]:
                total_batches = math.ceil(len(train_ds) / opts["batchSize"])
                print(
                    f"training: epoch {epoch:02d}: processing batch "
                    f"{batch_idx:03d} of {total_batches:03d} ... "
                    f"loss {loss.item():.6f}"
                )

        train_loss = train_loss_sum / max(train_count, 1)
        info["train"]["objective"].append(train_loss)
        info["train"]["error"].append(train_loss)

        if epoch == 1 or epoch % opts["test_interval"] == 0 or epoch == opts["numEpochs"]:
            net.eval()
            val_loss_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader, start=1):
                    x = _unpack_batch(batch).to(device).float()
                    x_hat = net(x)
                    loss = F.binary_cross_entropy(x_hat, x)

                    val_loss_sum += loss.item() * x.size(0)
                    val_count += x.size(0)

            val_loss = val_loss_sum / max(val_count, 1)
            info["val"]["objective"].append(val_loss)
            info["val"]["error"].append(val_loss)

            print(
                f"Epoch {epoch:03d}/{opts['numEpochs']} | "
                f"train={train_loss:.6f} | val={val_loss:.6f}"
            )

        if epoch % opts["snapshot"] == 0 or epoch == opts["numEpochs"]:
            ckpt_path = os.path.join(opts["expDir"], f"net-epoch-{epoch}.pt")
            _save_checkpoint(
                ckpt_path,
                net,
                optimizer,
                epoch,
                info,
                getattr(net, "input_size", None),
                getattr(net, "code_size", None)
            )

    return net, info
