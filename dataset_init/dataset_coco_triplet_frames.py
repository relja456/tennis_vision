import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetCocoBallMulti(Dataset):
    """
    COCO-style dataset for tennis ball tracking.
    Supports:
      - Legacy TrackNet (triplet, 9-channel stacked input, 1 heatmap)
      - TrackNetV4 (block of T frames, T heatmaps)

    root_dir/
      clip1/
        _annotations.coco.json
        0001.jpg ...
      clip2/
        _annotations.coco.json ...
    """

    def __init__(
        self,
        root_dir: str,
        out_size=(360, 640),
        t: int = 3,
        legacy_mode: bool = True,
        category_name: str = 'tennis-ball',
        category_id: int | None = None,
        heat_sigma: float = 2.5,
        annotations_filename: str = '_annotations.coco.json',
        image_exts=('.jpg', '.jpeg', '.png'),
        pick: str = 'first',  # when multiple boxes: "first"|"largest"|"smallest"
    ):
        self.root_dir = root_dir
        self.out_h, self.out_w = out_size
        self.T = t
        self.legacy_mode = legacy_mode
        self.heat_sigma = heat_sigma
        self.pick = pick
        self.image_exts = tuple(e.lower() for e in image_exts)

        self.clip_entries = []  # list of dicts: {"dir","frames","centers","orig_h","orig_w"}
        self.global_index = []

        for clip in sorted(os.listdir(root_dir)):
            clip_path = os.path.join(root_dir, clip)
            if not os.path.isdir(clip_path):
                continue
            ann_path = os.path.join(clip_path, annotations_filename)
            if not os.path.isfile(ann_path):
                continue

            with open(ann_path, encoding='utf-8') as f:
                coco = json.load(f)

            # resolve category
            if category_id is not None:
                cat_id = category_id
            else:
                names2id = {c['name']: c['id'] for c in coco.get('categories', [])}
                if category_name not in names2id:
                    continue
                cat_id = names2id[category_name]

            imgid2anns = {}
            for a in coco.get('annotations', []):
                imgid2anns.setdefault(a.get('image_id'), []).append(a)

            frames = []
            centers = {}
            for im in coco.get('images', []):
                fname = os.path.basename(im['file_name'])
                if not fname.lower().endswith(self.image_exts):
                    continue
                fpath = os.path.join(clip_path, fname)
                if not os.path.isfile(fpath):
                    continue
                frames.append(fname)

                cand = [a for a in imgid2anns.get(im['id'], []) if a.get('category_id') == cat_id]
                chosen = None
                if cand:
                    if self.pick in ('largest', 'smallest') and len(cand) > 1:
                        areas = [a.get('area', (a['bbox'][2] * a['bbox'][3]) if 'bbox' in a else 0.0) for a in cand]
                        idx = int(np.argmax(areas)) if self.pick == 'largest' else int(np.argmin(areas))
                        chosen = cand[idx]
                    else:
                        chosen = cand[0]

                if chosen and 'bbox' in chosen and len(chosen['bbox']) >= 4:
                    x, y, w, h = chosen['bbox']
                    cx = x + 0.5 * w
                    cy = y + 0.5 * h
                    centers[fname] = (float(cx), float(cy))
                else:
                    centers[fname] = None

            if not frames:
                continue
            frames.sort()

            probe = cv2.imread(os.path.join(clip_path, frames[0]), cv2.IMREAD_COLOR)
            if probe is None:
                continue
            orig_h, orig_w = probe.shape[:2]

            clip_idx = len(self.clip_entries)
            self.clip_entries.append({'dir': clip_path, 'frames': frames, 'centers': centers, 'orig_h': orig_h, 'orig_w': orig_w})
            for j in range(len(frames)):
                self.global_index.append((clip_idx, j))

        if not self.global_index:
            raise RuntimeError(f'No clips/images found under: {root_dir}')

        self.orig_h = self.clip_entries[0]['orig_h']
        self.orig_w = self.clip_entries[0]['orig_w']

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
        if cx is None or cy is None or cx < 0 or cy < 0:
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
            # --- Legacy TrackNet (triplet stacked, 1 heatmap) ---
            j0 = max(0, j - 1)
            j1 = j
            j2 = min(len(frames) - 1, j + 1)
            names = [frames[j0], frames[j1], frames[j2]]

            imgs = [self._read_rgb_resized(os.path.join(cdir, n)) for n in names]
            stacked = np.concatenate(imgs, axis=2)

            cxcy = entry['centers'].get(frames[j1], None)
            cx, cy = cxcy if cxcy else (None, None)
            if cx is not None and cy is not None:
                cx = cx * self.out_w / entry['orig_w']
                cy = cy * self.out_h / entry['orig_h']
            heat = self._make_gaussian_heatmap(self.out_h, self.out_w, cx, cy, sigma=self.heat_sigma)

            x = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0  # (9,H,W)
            y = torch.from_numpy(heat[None]).float()  # (1,H,W)
            return x, y

        else:
            # --- TrackNetV4 mode (T frames, T heatmaps) ---
            half = self.T // 2
            idxs = [min(max(k, 0), len(frames) - 1) for k in range(j - half, j - half + self.T)]

            imgs, heats = [], []
            for k in idxs:
                rgb = self._read_rgb_resized(os.path.join(cdir, frames[k]))
                imgs.append(rgb)

                cxcy = entry['centers'].get(frames[k], None)
                cx, cy = cxcy if cxcy else (None, None)
                if cx is not None and cy is not None:
                    cx = cx * self.out_w / entry['orig_w']
                    cy = cy * self.out_h / entry['orig_h']
                heat = self._make_gaussian_heatmap(self.out_h, self.out_w, cx, cy, sigma=self.heat_sigma)
                heats.append(heat)

            frames_t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
            targets_t = torch.from_numpy(np.stack(heats)[:, None]).float()  # (T,1,H,W)

            return {'frames': frames_t, 'targets': targets_t}
