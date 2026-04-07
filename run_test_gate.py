import torch

from config import FLOWERS_IMDB, BIRDS_IMDB, SCENES_IMDB, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE, FLOWERS_EXP, BIRDS_EXP, SCENES_EXP
from dataset_utils import build_dataset, build_loaders_from_dataset
from compute_relatedness import load_autoencoder_checkpoint
from test_triple_auto_gate import test_triple_auto_gate

def latest_ckpt(exp_dir, epoch=100):
    return f"{exp_dir}/net-epoch-{epoch}.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    flowers = build_dataset(FLOWERS_IMDB, IMAGENET_MEAN, IMAGENET_STD, apply_sigmoid=True)
    birds = build_dataset(BIRDS_IMDB, IMAGENET_MEAN, IMAGENET_STD, apply_sigmoid=True)
    scenes = build_dataset(SCENES_IMDB, IMAGENET_MEAN, IMAGENET_STD, apply_sigmoid=True)

    _, _, flowers_test = build_loaders_from_dataset(flowers, batch_size=BATCH_SIZE)
    _, _, birds_test = build_loaders_from_dataset(birds, batch_size=BATCH_SIZE)
    _, _, scenes_test = build_loaders_from_dataset(scenes, batch_size=BATCH_SIZE)

    encoders = [
        load_autoencoder_checkpoint(latest_ckpt(FLOWERS_EXP), device=device),
        load_autoencoder_checkpoint(latest_ckpt(BIRDS_EXP), device=device),
        load_autoencoder_checkpoint(latest_ckpt(SCENES_EXP), device=device),
    ]

    acc_all = test_triple_auto_gate(
        encoders=encoders,
        dataloaders_by_task=[flowers_test, birds_test, scenes_test],
        device=device
    )

    print("Final gate accuracies [Flowers, Birds, Scenes]:", acc_all)

if __name__ == "__main__":
    main()
