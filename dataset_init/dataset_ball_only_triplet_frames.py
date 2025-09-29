import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_labels_csv(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        labels = []
        next(reader, None)  # Skip the first row (header)
        for row in reader:
            if len(row) >= 5:
                visibility = float(row[1])
                if visibility == 0:
                    x, y, status = -1, -1, -1
                else:
                    x = float(row[2])
                    y = float(row[3])
                    status = int(row[4])
                labels.append({'visibility': visibility, 'x': x, 'y': y, 'status': status})
        return labels


class DatasetBallOnlyTripletFrames(Dataset):
    def __init__(self, root_dir, out_size=(360, 640), t=3, legacy_mode=True, sigma=5):
        self.root_dir = root_dir
        self.out_h, self.out_w = out_size
        self.orig_h, self.orig_w = out_size
        self.T = t
        self.legacy_mode = legacy_mode
        self.sigma = sigma

        self.clip_entries = []
        self.global_index = []

        for game in sorted(os.listdir(root_dir)):
            game_path = os.path.join(root_dir, game)

            if not os.path.isdir(game_path):
                continue

            for clip in sorted(os.listdir(game_path)):
                clip_path = os.path.join(game_path, clip)

                if not os.path.isdir(clip_path):
                    continue

                frames = [f for f in os.listdir(clip_path) if f.lower().endswith('.jpg')]

                if not frames:
                    continue

                frames.sort()

                csv_path = os.path.join(clip_path, 'Label.csv')

                if not os.path.isfile(csv_path):
                    raise FileNotFoundError(f'Missing Label.csv file in {clip_path}')

                labels = _read_labels_csv(csv_path)

                entry = {'dir': clip_path, 'frames': frames, 'labels': labels}

                clip_idx = len(self.clip_entries)
                self.clip_entries.append(entry)

                for j in range(len(frames)):
                    self.global_index.append((clip_idx, j))

        orig_img = cv2.imread(os.path.join(clip_path, '0000.jpg'))
        self.orig_h, self.orig_w, _ = orig_img.shape

        if not self.global_index:
            raise RuntimeError(f'No images found in root: {root_dir}')

    def __len__(self):
        return len(self.global_index)

    def _read_rgb_resized(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f'No file: {path}')
        bgr = cv2.resize(bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    @staticmethod
    def _make_gaussian_heatmap(h, w, cx, cy, sigma):
        if cx < 0 or cy < 0:
            return np.zeros((h, w), np.float32)
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)[:, None]
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        return g.astype(np.float32)

    def __getitem__(self, idx):
        clip_idx, j = self.global_index[idx]
        entry = self.clip_entries[clip_idx]
        frames = entry['frames']
        cdir = entry['dir']

        if self.legacy_mode:
            # --- Legacy TrackNet V1/V2/V3 style ---
            j0 = max(0, j - 1)
            j1 = j
            j2 = min(len(frames) - 1, j + 1)
            names = [frames[j0], frames[j1], frames[j2]]

            imgs = [self._read_rgb_resized(os.path.join(cdir, n)) for n in names]
            stacked = np.concatenate(imgs, axis=2)  # (H,W,9)

            cx, cy = entry['labels'][j]['x'], entry['labels'][j]['y']
            cx = min(self.orig_w - 1, max(0, cx))
            cy = min(self.orig_h - 1, max(0, cy))
            sx = cx * self.out_w / self.orig_w
            sy = cy * self.out_h / self.orig_h

            heat = self._make_gaussian_heatmap(self.out_h, self.out_w, sx, sy, self.sigma)

            x = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0  # (9,H,W)
            y = torch.from_numpy(heat[None]).float()  # (1,H,W)
            return x, y

        else:
            # --- TrackNetV4 style ---
            half = self.T // 2
            idxs = [min(max(k, 0), len(frames) - 1) for k in range(j - half, j - half + self.T)]

            imgs, heats = [], []
            for k in idxs:
                rgb = self._read_rgb_resized(os.path.join(cdir, frames[k]))
                imgs.append(rgb)

                cx, cy = entry['labels'][k]['x'], entry['labels'][k]['y']
                if cx < 0 or cy < 0 or entry['labels'][k]['visibility'] == 0:
                    heat = np.zeros((self.out_h, self.out_w), np.float32)
                else:
                    cx = min(self.orig_w - 1, max(0, cx))
                    cy = min(self.orig_h - 1, max(0, cy))
                    sx = cx * self.out_w / self.orig_w
                    sy = cy * self.out_h / self.orig_h
                    heat = self._make_gaussian_heatmap(self.out_h, self.out_w, sx, sy, self.sigma)
                heats.append(heat)

            frames_t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
            targets_t = torch.from_numpy(np.stack(heats)[:, None]).float()  # (T,1,H,W)

            return {'frames': frames_t, 'targets': targets_t}
