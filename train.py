import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from main import UNET
from SegNet import SegNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 512  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IMG_DIR = "data/train_images/"
# TRAIN_MASK_DIR = "data/train_masks/"
# VAL_IMG_DIR = "data/val_images/"
# VAL_MASK_DIR = "data/val_masks/"
TRAIN_IMG_DIR = "data/Images/"
TRAIN_MASK_DIR = "data/Masks/"
VAL_IMG_DIR = "data/Images_val/"
VAL_MASK_DIR = "data/Masks_val/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        torch.cuda.empty_cache()
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            #MOD
            #A.RandomCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            A.Rotate(limit=[-90,90], p=0.3),
            #MOD
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # MOD
            #A.RandomCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            # MOD
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    #model = SegNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer= optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8, momentum=0.9)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    #max_dice = -1.0
    #max_acc = -1.0
    max_TP = -1.0
    train_accs = []
    train_TPs = []
    train_TNs = []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}")
        torch.cuda.empty_cache()
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }


        # check accuracy
        dice, val_acc, TP, TN = check_accuracy(val_loader, model, device=DEVICE)

        dice1, train_acc, train_TP, train_TN = check_accuracy(train_loader, model, device=DEVICE)
        train_accs.append(train_acc/100)
        train_TPs.append(train_TP)
        train_TNs.append(train_TN)
        save_checkpoint(checkpoint, filename=f"my_checkpoint_{epoch+1}_{TP:.2f}_{val_acc:.2f}_{TN:.2f}.pth.tar") # was epoch
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, epoch+1, folder="saved_images/", device=DEVICE
        )
    plt.plot(train_accs, 'r')
    plt.plot(train_TPs, 'g')
    plt.plot(train_TNs, 'b')
    plt.title("Epoch progress")
    plt.ylabel("Measurements percentage")
    plt.xlabel("Epoch")
    plt.legend(['Accuracy', 'True Positive', 'True Negative'], loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
