import os

BASE_DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"

FLOWERS_IMDB = os.path.join(BASE_DATA_DIR, "flowers", "encoder_input_flowers_imdb.npz")
BIRDS_IMDB = os.path.join(BASE_DATA_DIR, "CUB", "encoder_input_cub_imdb.npz")
SCENES_IMDB = os.path.join(BASE_DATA_DIR, "scences", "encoder_input_scenes_imdb.npz")

IMAGENET_MEAN = os.path.join(BASE_DATA_DIR, "imagenet_mean.npy")
IMAGENET_STD = os.path.join(BASE_DATA_DIR, "imagenet_std.npy")

INPUT_SIZE = 43264
CODE_SIZE = 500
BATCH_SIZE = 12
NUM_EPOCHS = 100
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 5e-4

FLOWERS_EXP = os.path.join(CHECKPOINT_DIR, "Flowers", "autoencoder", "onelayer_test_relsig_std")
BIRDS_EXP = os.path.join(CHECKPOINT_DIR, "CUB_Training", "autoencoder", "onelayer_test_relsig_std")
SCENES_EXP = os.path.join(CHECKPOINT_DIR, "Scenes", "autoencoder", "onelayer_test_relsig_std")
