import os

import torch
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim
from tqdm.auto import tqdm


def conv3x3(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TrackNet(nn.Module):
    def __init__(self, in_channels=9, num_bins=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_bins = num_bins

        # ----- Encoder (VGG16-like)
        self.conv1_1 = conv3x3(in_channels, 64)  # Conv1 (64)
        self.conv1_2 = conv3x3(64, 64)  # Conv2 (64)
        self.pool1 = nn.MaxPool2d(2, 2)  # Pool1

        self.conv2_1 = conv3x3(64, 128)  # Conv3 (128)
        self.conv2_2 = conv3x3(128, 128)  # Conv4 (128)
        self.pool2 = nn.MaxPool2d(2, 2)  # Pool2

        self.conv3_1 = conv3x3(128, 256)  # Conv5 (256)
        self.conv3_2 = conv3x3(256, 256)  # Conv6 (256)
        self.conv3_3 = conv3x3(256, 256)  # Conv7 (256)
        self.pool3 = nn.MaxPool2d(2, 2)  # Pool3

        self.conv4_1 = conv3x3(256, 512)  # Conv8 (512)
        self.conv4_2 = conv3x3(512, 512)  # Conv9 (512)
        self.conv4_3 = conv3x3(512, 512)  # Conv10 (512)
        self.pool4 = nn.MaxPool2d(2, 2)  # Pool4

        self.conv5_1 = conv3x3(512, 512)  # Conv11 (512)
        self.conv5_2 = conv3x3(512, 512)  # Conv12 (512)
        self.conv5_3 = conv3x3(512, 512)  # Conv13 (512)

        # ----- Decoder (DeconvNet-like)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # Ups1
        self.deconv1_1 = conv3x3(512, 512)  # Conv14 (512)
        self.deconv1_2 = conv3x3(512, 512)  # Conv15 (512)
        self.deconv1_3 = conv3x3(512, 512)  # Conv16 (512)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # Ups2
        self.deconv2_1 = conv3x3(512, 256)  # Conv17 (256)
        self.deconv2_2 = conv3x3(256, 256)  # Conv18 (256)
        self.deconv2_3 = conv3x3(256, 256)  # Conv19 (256)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # Ups3
        self.deconv3_1 = conv3x3(256, 128)  # Conv20 (128)
        self.deconv3_2 = conv3x3(128, 128)  # Conv21 (128)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # Ups4
        self.deconv4_1 = conv3x3(128, 64)  # Conv22 (64)
        self.deconv4_2 = conv3x3(64, 64)  # Conv23 (64)

        # izlazna klasifikaciona mapa (softmax posle forward-a)
        self.classifier = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.optimizer = optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0], device=self.device))

    def forward(self, x):
        # ----- Encoder
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)  # 1/2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)  # 1/4
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)  # 1/8
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)  # 1/16
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)  # 1/16

        # ----- Decoder
        x = self.up1(x)
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv1_3(x)

        x = self.up2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = self.deconv2_3(x)

        x = self.up3(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)

        x = self.up4(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)

        logits = self.classifier(x)
        return logits

    def train(self, epochs=5, train_loader=None):
        best = 1e9
        torch.cuda.empty_cache()
        for epoch in range(1, epochs + 1):
            tr = self.train_one_epoch(train_loader)
            vl = self.evaluate()
            print(f"epoch {epoch:02d} | train {tr:.4f} | val {vl:.4f}")
            if vl < best:
                best = vl
                save_path = os.path.join(os.path.dirname(__file__), "tracknet_best.pth")
                torch.save({"model": self.state_dict()}, save_path)
                print(f"  ↳ saved {save_path}")

    def train_one_epoch(self, train_loader):
        self.train()
        total, n = 0.0, 0
        pbar = tqdm(train_loader, desc="Training", leave=False)

        self.optimizer.zero_grad(set_to_none=True)
        for xb, yb in pbar:
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = self(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear", align_corners=False)
                loss = self.criterion(logits, yb)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            total += loss.item() * xb.size(0)
            n += xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.scheduler.step()
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.eval()
        total, n = 0.0, 0
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = self(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear", align_corners=False)
                loss = self.criterion(logits, yb)

            total += loss.item() * xb.size(0)
            n += xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total / max(1, n)

    