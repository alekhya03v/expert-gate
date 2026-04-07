import torch
from config import SCENES_IMDB, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, CODE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SCENES_EXP
from dataset_utils import build_dataset, build_loaders_from_dataset
from cnn_autoencoder_layer_relusig import cnn_autoencoder_layer_relusig
from cnn_train_adagrad_oneLayer import cnn_train_adagrad_oneLayer

def main():
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(SCENES_IMDB, IMAGENET_MEAN, IMAGENET_STD, apply_sigmoid=True)
    train_loader, val_loader, _ = build_loaders_from_dataset(dataset, batch_size=BATCH_SIZE)

    net, _ = cnn_autoencoder_layer_relusig(
        input_size=INPUT_SIZE,
        code_size=CODE_SIZE,
        expDir=SCENES_EXP,
        useGpu=torch.cuda.is_available()
    )

    cnn_train_adagrad_oneLayer(
        net,
        train_loader,
        val_loader,
        numEpochs=NUM_EPOCHS,
        batchSize=BATCH_SIZE,
        learningRate=LEARNING_RATE,
        weightDecay=WEIGHT_DECAY,
        expDir=SCENES_EXP,
        useGpu=torch.cuda.is_available()
    )

if __name__ == "__main__":
    main()
