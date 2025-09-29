# TrackNetV4 — exact per-paper dataflow (PyTorch)
# - Input:  (B, T, 3, H, W)  short block of T' frames
# - Output: (B, T, 1, H, W)  T' heatmaps (logits; apply sigmoid outside)
# Matches: motion prompt (abs diff + PN) -> fusion (elem-mul then concat) -> head
# Citations: Eq. (1)-(4), Fig. 2–3 in arXiv:2409.14543. See chat.

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm


# ---------------------------
# Tiny VGG-like visual backbone (shared across frames)
# Produces high-level features V_t with C channels (C=64 by default)
# ---------------------------
def conv_bn_relu(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, 1, 1, bias=True),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class VisualBackbone(nn.Module):
    def __init__(self, in_ch=3, base=32, out_ch=64):
        super().__init__()
        # two downs, two ups -> full-res features (like TrackNet up to last conv)
        self.enc1 = nn.Sequential(conv_bn_relu(in_ch, base), conv_bn_relu(base, base))
        self.p1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(conv_bn_relu(base, base * 2), conv_bn_relu(base * 2, base * 2))
        self.p2 = nn.MaxPool2d(2)

        self.bott = nn.Sequential(conv_bn_relu(base * 2, base * 4), conv_bn_relu(base * 4, base * 4))

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = conv_bn_relu(base * 4, base * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = conv_bn_relu(base * 2, out_ch)  # final high-level features V_t

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B,3,H,W)
        x = self.enc1(x)
        x = self.p1(x)
        x = self.enc2(x)
        x = self.p2(x)
        x = self.bott(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)  # (B,C,H,W)
        return x


# ---------------------------
# Motion prompt: abs frame differencing + PN with 2 learnable params
# Produces A_t in [0,1] for t=0..T-2
# ---------------------------
class MotionPrompt(nn.Module):
    def __init__(self, slope_init=16.24, shift_init=0.28):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(slope_init)))
        self.shift = nn.Parameter(torch.tensor(float(shift_init)))

    @staticmethod
    def rgb_to_gray(frames_btc3hw: torch.Tensor) -> torch.Tensor:
        # (B,T,3,H,W) -> (B,T,H,W)
        r, g, b = frames_btc3hw[:, :, 0], frames_btc3hw[:, :, 1], frames_btc3hw[:, :, 2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(0.0, 1.0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B,T,3,H,W) in [0,1]
        B, T, _, H, W = frames.shape
        gray = self.rgb_to_gray(frames)  # (B,T,H,W)
        diffs = (gray[:, 1:] - gray[:, :-1]).abs()  # (B,T-1,H,W)  --> D+_t  (abs differencing)
        A = torch.sigmoid(self.slope * (diffs - self.shift))  # PN with 2 learnable params, in [0,1]
        return A  # (B,T-1,H,W)


# ---------------------------
# TrackNetV4 — per paper
# ---------------------------
class TrackNetV4(nn.Module):
    def __init__(self, feat_ch=64):
        super().__init__()
        self.visual = VisualBackbone(out_ch=feat_ch)  # TrackNetvisual(·)
        self.motion = MotionPrompt()
        # Fusion "⊚": [ V_t , A_t ⊙ V_{t+1} ]  -> 1×1 conv -> logits
        self.head = nn.Conv2d(feat_ch * 2, 1, kernel_size=1)

        nn.init.kaiming_normal_(self.head.weight, nonlinearity='linear')
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x_btc3hw: torch.Tensor) -> torch.Tensor:
        """
        x_btc3hw: (B, T, 3, H, W)  short temporal block
        returns:  (B, T, 1, H, W)  logits for each of the T frames
        """
        B, T, C, H, W = x_btc3hw.shape
        # Visual features per frame (shared weights)
        V = []
        for t in range(T):
            V.append(self.visual(x_btc3hw[:, t]))  # each: (B,Cv,H,W)
        V = torch.stack(V, dim=1)  # (B,T,Cv,H,W)

        # Motion attentions for t=0..T-2
        A = self.motion(x_btc3hw)  # (B,T-1,H,W)

        # Fusion & heads -> logits per t (Eq. 3 & 4)
        logits = []
        for t in range(T):
            Vt = V[:, t]  # (B,Cv,H,W)
            if t < T - 1:
                At = A[:, t].unsqueeze(1)  # (B,1,H,W)
                Zt = torch.cat([Vt, V[:, t + 1] * At], dim=1)  # [V_t, A_t ⊙ V_{t+1}]
            else:
                # Last step: no A_{T-1}; pad with zeros to keep channels = 2*Cv
                Zt = torch.cat([Vt, torch.zeros_like(Vt)], dim=1)
            logits.append(self.head(Zt))  # (B,1,H,W)

        return torch.stack(logits, dim=1)  # (B,T,1,H,W)

    def train_override(self, epochs=50, train_loader=None, val_loader=None, save_name='tracknetv4_best.pth'):
        best = float('inf')
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        for epoch in range(1, epochs + 1):
            tr = self.train_one_epoch(train_loader)
            vl = self.evaluate(val_loader)
            print(f'epoch {epoch:02d} | train {tr:.4f} | val {vl:.4f}')
            if vl < best:
                best = vl
                save_path = os.path.join(os.path.dirname(__file__), save_name)
                torch.save({'model': self.model.state_dict()}, save_path)
                print(f'  ↳ saved {save_path}')

    def train_one_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc='Training', leave=False)

        self.optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            x = batch['frames'].to(self.device, non_blocking=True)  # (B,T,3,H,W)
            y = batch['targets'].to(self.device, non_blocking=True)  # (B,T,1,H,W)

            with torch.autocast(
                device_type=('cuda' if self.device == 'cuda' else 'cpu'), dtype=torch.float16, enabled=(self.device == 'cuda')
            ):
                logits = self.model(x)  # (B,T,1,H,W)
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            # (opciono) grad clipping:
            # self.scaler.unscale_(self.optimizer); nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            total += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        if self.scheduler is not None:
            self.scheduler.step()

        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            x = batch['frames'].to(self.device, non_blocking=True)
            y = batch['targets'].to(self.device, non_blocking=True)

            logits = self.model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

            loss = self.criterion(logits, y)
            total += loss.item() * x.size(0)
            n += x.size(0)
        return total / max(1, n)


# quick sanity check
if __name__ == '__main__':
    B, T, H, W = 2, 3, 360, 640
    x = torch.rand(B, T, 3, H, W)
    model = TrackNetV4(feat_ch=64)
    y = model(x)
    print('logits:', tuple(y.shape))  # (B, T, 1, H, W)
