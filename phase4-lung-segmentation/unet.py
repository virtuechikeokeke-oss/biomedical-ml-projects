import torch
import torch.nn as nn

# ── building block ─────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """
    Two consecutive Conv → BatchNorm → ReLU blocks.
    This is the repeated unit used at every level of the U-Net.
    Why double? Two convolutions give the network more capacity to learn
    features at each resolution level before moving to the next.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # first conv: learn features from input
            # kernel_size=3 looks at a 3x3 neighborhood of pixels
            # padding=1 keeps spatial dimensions the same (same padding)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # batch norm: normalizes activations across the batch
            # stabilizes training — without it, deep networks are hard to train
            nn.BatchNorm2d(out_channels),
            # ReLU: non-linearity — sets negative values to 0
            # without non-linearities, stacking layers does nothing (still linear)
            nn.ReLU(inplace=True),

            # second conv: refine the features learned by the first conv
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── encoder block ──────────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    """
    One step down the U-Net encoder (left side).
    DoubleConv extracts features, then MaxPool halves spatial dimensions.
    We return the pre-pool features as the skip connection — this is the
    'photograph at this zoom level' that gets passed to the decoder later.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        # MaxPool2d reduces H and W by half (2x2 window, stride 2)
        # this is the "zooming out" step — loses spatial detail, gains context
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # extract features at current resolution — save this for skip connection
        features = self.conv(x)
        # downsample for next encoder level
        pooled = self.pool(features)
        # return both: pooled goes deeper, features goes sideways to decoder
        return features, pooled


# ── decoder block ──────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    """
    One step up the U-Net decoder (right side).
    Upsamples spatial dimensions, then concatenates the skip connection
    from the matching encoder level, then applies DoubleConv.
    The concatenation is where the skip connection is actually used —
    it doubles the channel count, giving the decoder both:
      - low-res context from the decoder path
      - high-res detail from the encoder path
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d is the learnable upsampling operation
        # doubles H and W — the reverse of MaxPool
        # in_channels → in_channels//2 because after upsampling we will
        # concatenate skip features which brings channels back up
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
        # after concat, channels = in_channels//2 (upsampled) + in_channels//2 (skip)
        # = in_channels total → DoubleConv reduces this to out_channels
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        # upsample: double spatial dimensions
        x = self.upsample(x)
        # concatenate skip connection along channel dimension (dim=1)
        # x shape:    [B, C/2, H, W]
        # skip shape: [B, C/2, H, W]
        # after cat:  [B, C,   H, W]
        x = torch.cat([skip, x], dim=1)
        # refine the combined features
        return self.conv(x)


# ── full u-net ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    Full U-Net for binary segmentation.
    Architecture:
      Encoder: in_channels → 64 → 128 → 256 → 512
      Bottleneck:            512 → 1024
      Decoder:  1024 → 512 → 256 → 128 → 64
      Output:    64 → 1  (binary mask, one channel)

    Input:  [B, in_channels, H, W]
    Output: [B, 1, H, W] — batch of predicted masks (logits)
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # ── encoder (contracting path) ─────────────────────────────────
        # each block halves H,W and doubles channels
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # ── bottleneck ─────────────────────────────────────────────────
        # deepest point — most compressed spatial representation
        # no pooling after this, just feature extraction
        self.bottleneck = DoubleConv(512, 1024)

        # ── decoder (expanding path) ───────────────────────────────────
        # each block doubles H,W and halves channels
        # receives skip connection from matching encoder level
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # ── output layer ───────────────────────────────────────────────
        # 1x1 conv maps 64 channels → 1 channel (binary mask)
        # no sigmoid here — raw logits are better for numerical stability
        # sigmoid is applied inside the loss function (BCEWithLogitsLoss)
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # ── encoder ────────────────────────────────────────────────────
        # each returns (skip_features, pooled_for_next_level)
        skip1, x = self.enc1(x)   # skip1: [B,64,H/1,W/1]   x: [B,64,H/2,W/2]
        skip2, x = self.enc2(x)   # skip2: [B,128,H/2,W/2]  x: [B,128,H/4,W/4]
        skip3, x = self.enc3(x)   # skip3: [B,256,H/4,W/4]  x: [B,256,H/8,W/8]
        skip4, x = self.enc4(x)   # skip4: [B,512,H/8,W/8]  x: [B,512,H/16,W/16]

        # ── bottleneck ─────────────────────────────────────────────────
        x = self.bottleneck(x)    # x: [B,1024,H/16,W/16]

        # ── decoder ────────────────────────────────────────────────────
        # pass current x AND the matching skip from the encoder
        x = self.dec4(x, skip4)   # x: [B,512,H/8,W/8]
        x = self.dec3(x, skip3)   # x: [B,256,H/4,W/4]
        x = self.dec2(x, skip2)   # x: [B,128,H/2,W/2]
        x = self.dec1(x, skip1)   # x: [B,64,H,W]

        # ── output ─────────────────────────────────────────────────────
        return self.output_conv(x)  # x: [B,1,H,W] — raw logits