import torch
import torch.nn as nn

# ── loss functions ─────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss — measures overlap between predicted mask and ground truth.
    Range: 0 (perfect overlap) to 1 (no overlap).
    Formula: 1 - (2 * |pred ∩ true|) / (|pred| + |true|)
    
    Why we need it alongside BCE:
    BCE grades every pixel equally — on imbalanced data (lots of background,
    little foreground) the model learns to predict all background and still
    gets a great BCE score. Dice only cares about overlap ratio so it forces
    the model to actually find and predict the foreground region.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        # smooth (epsilon) prevents division by zero when both pred and
        # target are all zeros — adds a tiny constant to numerator and denominator
        self.smooth = smooth

    def forward(self, logits, targets):
        # apply sigmoid to convert raw logits → probabilities [0,1]
        # we do this here rather than in the model for numerical stability
        preds = torch.sigmoid(logits)

        # flatten spatial dimensions so we compute one scalar per image
        # shape: [B, 1, H, W] → [B, H*W]
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # intersection: where both pred and target are high (both close to 1)
        # element-wise multiply then sum — only contributes when both are 1
        intersection = (preds * targets).sum(dim=1)

        # dice score per image in the batch
        dice_score = (2.0 * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        # average dice score across batch, subtract from 1 to make it a loss
        # (loss should go DOWN as model improves, but dice score goes UP)
        return 1.0 - dice_score.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss — standard for medical/biological segmentation.
    
    BCE catches per-pixel errors precisely.
    Dice handles class imbalance by focusing on overlap ratio.
    Together they're more robust than either alone:
    - BCE provides stable gradients early in training
    - Dice keeps the model honest about actually finding the foreground
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        # BCEWithLogitsLoss = sigmoid + BCE in one numerically stable operation
        # more stable than applying sigmoid manually then using BCELoss
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        # weights control how much each loss contributes to the total
        # 0.5/0.5 treats them equally — can be tuned
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        # weighted sum — both losses are on similar scales so 0.5/0.5 works well
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ── evaluation metric ──────────────────────────────────────────────────────

def iou_score(logits, targets, threshold=0.5):
    """
    Intersection over Union (IoU) — the standard segmentation metric.
    Formula: |pred ∩ true| / |pred ∪ true|
    Range: 0 (no overlap) to 1 (perfect overlap).
    
    Plain version: imagine two circles drawn on paper.
    IoU = (area where they overlap) / (total area covered by both circles).
    Perfect prediction = both circles are identical = IoU of 1.0.
    
    We use threshold=0.5: sigmoid output > 0.5 → predicted foreground,
    anything below → predicted background. Converts probabilities to binary.
    """
    # convert logits → binary predictions
    # sigmoid: raw score → probability, then threshold to 0 or 1
    preds = (torch.sigmoid(logits) > threshold).float()

    # flatten: [B, 1, H, W] → [B, H*W]
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # intersection: pixels predicted foreground AND actually foreground
    intersection = (preds * targets).sum(dim=1)

    # union: pixels predicted foreground OR actually foreground
    # union = pred + target - intersection (avoid double counting overlap)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    # add smooth to avoid division by zero on empty masks
    iou = (intersection + 1e-6) / (union + 1e-6)

    # return mean IoU across the batch
    return iou.mean().item()