import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloaders
from unet import UNet
from utils import BCEDiceLoss, iou_score

# ── config ─────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if available
EPOCHS     = 30       # number of full passes through the training data
LR         = 1e-4     # learning rate — how big each gradient update step is
SAVE_PATH  = "/teamspace/studios/this_studio/phase4-segmentation/best_model.pth"

# ── training loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn):
    """
    Runs one full pass through the training data.
    Updates model weights after every batch via backpropagation.
    Returns average loss and IoU across all batches.
    """
    # set model to training mode — enables dropout and batch norm updates
    # (we don't use dropout here but it's good practice to always set this)
    model.train()
    total_loss = 0.0
    total_iou  = 0.0

    # tqdm wraps the loader to show a live progress bar in the terminal
    for images, masks in tqdm(loader, desc="Training", leave=False):
        # move data to GPU (or CPU if no GPU available)
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        # ── forward pass ────────────────────────────────────────────
        # feed images through U-Net to get predicted mask logits
        predictions = model(images)

        # compute how wrong the predictions are vs ground truth masks
        loss = loss_fn(predictions, masks)

        # ── backward pass ───────────────────────────────────────────
        # zero_grad: clear gradients from the previous batch
        # if we don't clear them, they accumulate and corrupt the update
        optimizer.zero_grad()

        # backprop: compute gradient of loss with respect to every weight
        # this is PyTorch's autograd doing the chain rule automatically
        loss.backward()

        # update weights: take one step in the direction that reduces loss
        optimizer.step()

        # accumulate metrics for reporting
        total_loss += loss.item()
        total_iou  += iou_score(predictions, masks)

    # return averages across all batches
    avg_loss = total_loss / len(loader)
    avg_iou  = total_iou  / len(loader)
    return avg_loss, avg_iou


# ── validation loop ────────────────────────────────────────────────────────
def validate(model, loader, loss_fn):
    """
    Runs one full pass through the validation data.
    No weight updates — just measuring how well the model generalises
    to images it hasn't been trained on.
    """
    # eval mode: freezes batch norm running stats, disables dropout
    # critical — forgetting this gives you artificially good val metrics
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0

    # torch.no_grad: disables gradient computation entirely
    # validation doesn't need gradients — saves memory and speeds things up
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            predictions = model(images)
            loss = loss_fn(predictions, masks)

            total_loss += loss.item()
            total_iou  += iou_score(predictions, masks)

    avg_loss = total_loss / len(loader)
    avg_iou  = total_iou  / len(loader)
    return avg_loss, avg_iou


# ── visualisation ──────────────────────────────────────────────────────────
def visualise_predictions(model, loader, num_samples=4):
    """
    Shows side-by-side comparison of input image, ground truth mask,
    and model prediction for a few samples.
    Lets you visually inspect what the model is actually learning.
    """
    model.eval()
    images, masks = next(iter(loader))  # grab one batch
    images, masks = images.to(DEVICE), masks.to(DEVICE)

    with torch.no_grad():
        predictions = model(images)
        # sigmoid → probabilities, threshold at 0.5 → binary mask
        pred_masks = (torch.sigmoid(predictions) > 0.5).float()

    # plot num_samples rows, 3 columns: image | ground truth | prediction
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    axes[0, 0].set_title("Input Image",    fontsize=12)
    axes[0, 1].set_title("Ground Truth",   fontsize=12)
    axes[0, 2].set_title("Prediction",     fontsize=12)

    for i in range(num_samples):
        # denormalise image back to [0,1] for display
        # we reverse the ImageNet normalisation applied in dataset.py
        # grayscale image: undo normalization (mean=0.5, std=0.5)
        img = images[i, 0].cpu().numpy()   # [H, W] — drop channel dim for display
        img = img * 0.5 + 0.5              # reverse normalize back to [0,1]
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 1].imshow(masks[i, 0].cpu(), cmap="gray")
        axes[i, 2].imshow(pred_masks[i, 0].cpu(), cmap="gray")

        for ax in axes[i]:
            ax.axis("off")  # remove axis ticks for cleaner display

    plt.tight_layout()
    plt.savefig("/teamspace/studios/this_studio/phase4-segmentation/predictions.png", dpi=150)
    plt.close()
    print("Saved predictions.png")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    # load data
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # initialise model, loss, optimiser
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn  = BCEDiceLoss()
    # Adam: adaptive learning rate optimiser — standard choice for segmentation
    # adjusts the learning rate per parameter based on gradient history
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # learning rate scheduler: reduces LR by half if val loss stops improving
    # for 3 epochs straight — prevents overshooting the minimum
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ── training loop ───────────────────────────────────────────────
    best_val_loss = float("inf")  # track best validation loss for model saving
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss,   val_iou   = validate(model, val_loader, loss_fn)

        # step scheduler with val loss — may reduce LR if plateau detected
        scheduler.step(val_loss)

        # save model whenever validation loss improves
        # "best model" = weights at the epoch with lowest val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            saved_str = "  ← saved"
        else:
            saved_str = ""

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}  IoU: {val_iou:.4f}{saved_str}")

        # store for plotting
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

    # ── post training ───────────────────────────────────────────────
    print("\nTraining complete. Loading best model for evaluation...")
    model.load_state_dict(torch.load(SAVE_PATH))

    test_loss, test_iou = validate(model, test_loader, loss_fn)
    print(f"Test Loss: {test_loss:.4f} | Test IoU: {test_iou:.4f}")

    # plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss"); ax1.legend()

    ax2.plot(history["train_iou"], label="Train IoU")
    ax2.plot(history["val_iou"],   label="Val IoU")
    ax2.set_title("IoU"); ax2.legend()

    plt.savefig("/teamspace/studios/this_studio/phase4-segmentation/training_curves.png", dpi=150)
    plt.close()
    print("Saved training_curves.png")

    # visualise some predictions
    visualise_predictions(model, test_loader)


if __name__ == "__main__":
    main()