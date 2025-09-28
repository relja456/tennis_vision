import csv
import math
import os
import pathlib

from matplotlib import pyplot as plt
import numpy as np
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Ups1
        self.deconv1_1 = conv3x3(512, 512)  # Conv14 (512)
        self.deconv1_2 = conv3x3(512, 512)  # Conv15 (512)
        self.deconv1_3 = conv3x3(512, 512)  # Conv16 (512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Ups2
        self.deconv2_1 = conv3x3(512, 256)  # Conv17 (256)
        self.deconv2_2 = conv3x3(256, 256)  # Conv18 (256)
        self.deconv2_3 = conv3x3(256, 256)  # Conv19 (256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Ups3
        self.deconv3_1 = conv3x3(256, 128)  # Conv20 (128)
        self.deconv3_2 = conv3x3(128, 128)  # Conv21 (128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Ups4
        self.deconv4_1 = conv3x3(128, 64)  # Conv22 (64)
        self.deconv4_2 = conv3x3(64, 64)  # Conv23 (64)

        # izlazna klasifikaciona mapa (softmax posle forward-a)
        self.classifier = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.optimizer = optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=self.device))

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

    def train_override(self, epochs=5, train_loader=None, val_loader=None):
        best = 1e9
        torch.cuda.empty_cache()
        for epoch in range(1, epochs + 1):
            tr = self.train_one_epoch(train_loader)
            vl = self.evaluate(val_loader)
            print(f'epoch {epoch:02d} | train {tr:.4f} | val {vl:.4f}')
            if vl < best:
                best = vl
                save_path = os.path.join(os.path.dirname(__file__), 'tracknet_best.pth')
                torch.save({'model': self.state_dict()}, save_path)
                print(f'  ↳ saved {save_path}')

    def train_one_epoch(self, train_loader):
        self.train()
        total, n = 0.0, 0
        pbar = tqdm(train_loader, desc='Training', leave=False)

        self.optimizer.zero_grad(set_to_none=True)
        for xb, yb in pbar:
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = self(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode='bilinear', align_corners=False)
                loss = self.criterion(logits, yb)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            total += loss.item() * xb.size(0)
            n += xb.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        self.scheduler.step()
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.eval()
        total, n = 0.0, 0
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = self(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode='bilinear', align_corners=False)
                loss = self.criterion(logits, yb)

            total += loss.item() * xb.size(0)
            n += xb.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        return total / max(1, n)

    @staticmethod
    def _argmax2d(t: torch.Tensor):
        """
        t: (H,W) tensor -> returns (y, x, val)
        """
        _, w = t.shape
        idx = torch.argmax(t)
        y = (idx // w).item()
        x = (idx % w).item()
        return y, x, t[y, x].item()

    @staticmethod
    def _euclid(x1, y1, x2, y2):
        dx = float(x1) - float(x2)
        dy = float(y1) - float(y2)
        return math.hypot(dx, dy)

    def _peak_from_logits(self, logits: torch.Tensor, out_hw=None):
        """
        logits: (B,1,h,w) -> prob (B,1,h,w), peak (x,y,val) per item
        If out_hw is given, resize to out_hw first.
        """
        if out_hw is not None and logits.shape[-2:] != out_hw:
            logits = F.interpolate(logits, size=out_hw, mode='bilinear', align_corners=False)
        prob = torch.sigmoid(logits)
        peaks = []
        for i in range(prob.size(0)):
            y, x, v = self._argmax2d(prob[i, 0])
            peaks.append((x, y, v))
        return prob, peaks

    def _gt_center_from_heat(self, heat: torch.Tensor):
        """
        heat: (B,1,H,W) target heatmaps in [0,1]
        Returns list of (x,y) or None (if no GT ball in frame).
        """
        centers = []
        for i in range(heat.size(0)):
            y, x, v = self._argmax2d(heat[i, 0])
            if v >= self.GT_MIN:
                centers.append((x, y))
            else:
                centers.append(None)
        return centers

    @torch.no_grad()
    def evaluate_tracknet_metrics(self, loader, save_dir=None, epoch=None):
        """
        Computes TrackNet-paper metrics:
          - Precision, Recall, F1  (TP if detected & PE <= PE_TOL)
          - Positioning Error stats: mean/median/std and % within PE_TOL (on frames with GT)
        Also returns average BCE loss for reference and optionally saves CSV & plots.
        """
        self.eval()
        total_loss, n_loss = 0.0, 0

        tp = fp = fn = 0
        pe_list = []  # PE on frames where GT exists and we produced a detection
        within_tol = 0  # count PE <= PE_TOL among (GT exists) frames
        gt_frames = 0  # frames with GT ball

        # to track loss curve (optional aggregation per epoch only)
        for xb, yb in tqdm(loader, desc='Metrics (val)', leave=False):
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)  # (B,1,H,W) target heat

            # forward
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = self(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode='bilinear', align_corners=False)
                loss = self.criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            n_loss += xb.size(0)

            # predicted peaks & GT centers
            _prob, pred_peaks = self._peak_from_logits(logits, out_hw=yb.shape[-2:])
            gt_centers = self._gt_center_from_heat(yb)

            # per frame decisions
            for i in range(xb.size(0)):
                px, py, pval = pred_peaks[i]
                gt = gt_centers[i]

                detected = pval >= self.DET_THRESH
                has_gt = gt is not None
                if has_gt:
                    gt_frames += 1

                if detected and has_gt:
                    pe = self._euclid(px, py, gt[0], gt[1])
                    pe_list.append(pe)
                    if pe <= self.PE_TOL:
                        tp += 1
                        within_tol += 1
                    else:
                        fp += 1
                elif detected and not has_gt:
                    fp += 1
                elif (not detected) and has_gt:
                    fn += 1
                # if neither detected nor has_gt => ignore (no TN in paper's headline metrics)

        # aggregate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        # PE stats (only frames with GT & detection)
        if len(pe_list) > 0:
            pe_mean = float(np.mean(pe_list))
            pe_median = float(np.median(pe_list))
            pe_std = float(np.std(pe_list))
        else:
            pe_mean = pe_median = pe_std = float('nan')

        pct_within_tol = (within_tol / gt_frames) if gt_frames > 0 else 0.0

        avg_loss = total_loss / max(1, n_loss)

        results = {
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_frames': gt_frames,
            'pe_mean': pe_mean,
            'pe_median': pe_median,
            'pe_std': pe_std,
            f'pct_within_{int(self.PE_TOL)}px': pct_within_tol,
        }

        # optional save
        if save_dir is not None:
            self._save_metrics_artifacts(results, pe_list, save_dir, epoch)

        return results

    # ------------- artifact saving (CSV + plots) -------------
    def _save_metrics_artifacts(self, results: dict, pe_list, save_dir, epoch=None):
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # CSV (append)
        csv_path = save_dir / 'metrics_log.csv'
        header = [
            'epoch',
            'loss',
            'precision',
            'recall',
            'f1',
            'tp',
            'fp',
            'fn',
            'gt_frames',
            'pe_mean',
            'pe_median',
            'pe_std',
            f'pct_within_{int(self.PE_TOL)}px',
        ]
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(
                [
                    epoch if epoch is not None else -1,
                    f'{results["loss"]:.6f}',
                    f'{results["precision"]:.6f}',
                    f'{results["recall"]:.6f}',
                    f'{results["f1"]:.6f}',
                    results['tp'],
                    results['fp'],
                    results['fn'],
                    results['gt_frames'],
                    f'{results["pe_mean"]:.6f}' if not math.isnan(results['pe_mean']) else 'nan',
                    f'{results["pe_median"]:.6f}' if not math.isnan(results['pe_median']) else 'nan',
                    f'{results["pe_std"]:.6f}' if not math.isnan(results['pe_std']) else 'nan',
                    f'{results[f"pct_within_{int(self.PE_TOL)}px"]:.6f}',
                ]
            )

        # Loss trend plot: append point to PNG (quick redraw per epoch)
        # Here we simply redraw from CSV for simplicity
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            fig = plt.figure()
            plt.plot(df['epoch'], df['loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Loss')
            fig.tight_layout()
            fig.savefig(save_dir / 'loss_curve.png', dpi=150)
            plt.close(fig)
        except Exception:
            pass

        # PE histogram (only if we have samples)
        if len(pe_list) > 0:
            fig = plt.figure()
            plt.hist(pe_list, bins=40)
            plt.axvline(self.PE_TOL, linestyle='--')
            plt.xlabel('Positioning Error (px)')
            plt.ylabel('Count')
            ttl = f'PE Histogram (epoch {epoch})' if epoch is not None else 'PE Histogram'
            plt.title(ttl)
            fig.tight_layout()
            fig.savefig(save_dir / (f'pe_hist_epoch_{epoch}.png' if epoch is not None else 'pe_hist.png'), dpi=150)
            plt.close(fig)
