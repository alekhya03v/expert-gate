import torch
from torch.utils.data import DataLoader, Subset

from config import FLOWERS_IMDB, BIRDS_IMDB, SCENES_IMDB, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE, FLOWERS_EXP, BIRDS_EXP, SCENES_EXP
from dataset_utils import build_dataset, split_indices_from_sets
from compute_relatedness import load_autoencoder_checkpoint, compute_relatedness

def test_loader_from_dataset(dataset, batch_size):
    _, _, test_idx = split_indices_from_sets(dataset.sets)
    test_ds = Subset(dataset, test_idx.tolist())
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)

def latest_ckpt(exp_dir, epoch=100):
    return f"{exp_dir}/net-epoch-{epoch}.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    flowers = build_dataset(FLOWERS_IMDB, IMAGENET_MEAN, IMAGENET_STD, apply_sigmoid=True)
    birds = build_dataset(BIRDS_IMDB, IMAGENET_MEAN, IMAGENET_STD, apply_sigmoid=True)
    scenes = build_dataset(SCENES_IMDB, IMAGENET_MEAN, IMAGENET_STD, apply_sigmoid=True)

    flowers_enc = load_autoencoder_checkpoint(latest_ckpt(FLOWERS_EXP), device=device)
    birds_enc = load_autoencoder_checkpoint(latest_ckpt(BIRDS_EXP), device=device)
    scenes_enc = load_autoencoder_checkpoint(latest_ckpt(SCENES_EXP), device=device)

    print("\nFlowers vs Birds-on-Birds:")
    print(compute_relatedness(flowers_enc, birds_enc, test_loader_from_dataset(birds, BATCH_SIZE), device=device))

    print("\nFlowers vs Scenes-on-Scenes:")
    print(compute_relatedness(flowers_enc, scenes_enc, test_loader_from_dataset(scenes, BATCH_SIZE), device=device))

    print("\nBirds vs Scenes-on-Scenes:")
    print(compute_relatedness(birds_enc, scenes_enc, test_loader_from_dataset(scenes, BATCH_SIZE), device=device))

if __name__ == "__main__":
    main()
