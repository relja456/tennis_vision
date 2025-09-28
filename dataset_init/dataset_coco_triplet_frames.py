import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetCocoBallTripletMulti(Dataset):
    """
    Multi-clip COCO dataset:
      root_dir/
        clip1/
          _annotations.coco.json
          0001.jpg
          0003.jpg
          ...
        clip2/
          _annotations.coco.json
          ...

    Returns triplets (t-1, t, t+1) stacked into 9-channel tensor and a 1-channel heatmap
    centered at the bbox (for the target category) of the middle frame.
    """

    def __init__(
        self,
        root_dir: str,
        out_size=(360, 640),
        t: int = 3,
        category_name: str = 'tennis-ball',
        category_id: int | None = None,  # if given, overrides category_name lookup
        heat_sigma: float = 2.5,
        annotations_filename: str = '_annotations.coco.json',
        image_exts=('.jpg', '.jpeg', '.png'),
        pick: str = 'first',  # when multiple boxes: "first" | "largest" | "smallest"
    ):
        assert t == 3, 'This dataset constructs triplets; set t=3.'
        self.root_dir = root_dir
        self.out_h, self.out_w = out_size
        self.T = t
        self.heat_sigma = heat_sigma
        self.pick = pick
        self.image_exts = tuple(e.lower() for e in image_exts)

        self.clip_entries = []  # list of dicts: {"dir", "frames", "centers", "orig_h", "orig_w"}
        self.global_index = []  # list of (clip_idx, j)

        # build from each clip folder
        for clip in sorted(os.listdir(root_dir)):
            clip_path = os.path.join(root_dir, clip)
            if not os.path.isdir(clip_path):
                continue

            ann_path = os.path.join(clip_path, annotations_filename)
            if not os.path.isfile(ann_path):
                # skip folders without annotations json
                continue

            with open(ann_path, encoding='utf-8') as f:
                coco = json.load(f)

            # resolve target category id
            if category_id is not None:
                cat_id = category_id
            else:
                names2id = {c['name']: c['id'] for c in coco.get('categories', [])}
                if category_name not in names2id:
                    # no such category in this clip; skip it
                    continue
                cat_id = names2id[category_name]

            # index anns by image_id
            imgid2anns = {}
            for a in coco.get('annotations', []):
                imgid2anns.setdefault(a.get('image_id'), []).append(a)

            # collect frames present on disk, matched by basename(file_name)
            frames = []
            centers = {}  # basename -> (cx, cy) in ORIGINAL pixel coords, or None
            for im in coco.get('images', []):
                fname = os.path.basename(im['file_name'])
                if not fname.lower().endswith(self.image_exts):
                    continue
                fpath = os.path.join(clip_path, fname)
                if not os.path.isfile(fpath):
                    continue
                frames.append(fname)

                # choose annotation for this image (if any) for the target category
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
                    centers[fname] = None  # no target => zero heatmap

            if not frames:
                continue

            frames.sort()

            # probe original size from first existing frame
            probe = cv2.imread(os.path.join(clip_path, frames[0]), cv2.IMREAD_COLOR)
            if probe is None:
                continue
            orig_h, orig_w = probe.shape[:2]

            # store entry
            clip_idx = len(self.clip_entries)
            self.clip_entries.append(
                {
                    'dir': clip_path,
                    'frames': frames,
                    'centers': centers,
                    'orig_h': orig_h,
                    'orig_w': orig_w,
                }
            )

            # expand global index
            for j in range(len(frames)):
                self.global_index.append((clip_idx, j))

        if not self.global_index:
            raise RuntimeError(f'No clips/images found under: {root_dir}')

        # expose "global" orig size (from first clip) for downstream code that references dataset.orig_h/w
        self.orig_h = self.clip_entries[0]['orig_h']
        self.orig_w = self.clip_entries[0]['orig_w']

    def __len__(self):
        return len(self.global_index)

    # ---------- helpers ----------
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

        # triplet indices
        j0 = max(0, j - 1)
        j1 = j
        j2 = min(len(frames) - 1, j + 1)
        names = [frames[j0], frames[j1], frames[j2]]

        # read + stack -> (H,W,9)
        imgs = [self._read_rgb_resized(os.path.join(cdir, n)) for n in names]
        stacked = np.concatenate(imgs, axis=2)

        # center for current (middle) frame, scaled to out_size
        cxcy = entry['centers'].get(frames[j1], None)
        if cxcy is None:
            cx, cy = None, None
        else:
            cx, cy = cxcy
            # clamp to original size for safety
            cx = min(max(0.0, cx), float(entry['orig_w']))
            cy = min(max(0.0, cy), float(entry['orig_h']))
            # scale to out_size
            cx = cx * self.out_w / entry['orig_w']
            cy = cy * self.out_h / entry['orig_h']

        heat = self._make_gaussian_heatmap(self.out_h, self.out_w, cx, cy, sigma=self.heat_sigma)

        x = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0  # (9,H,W)
        y = torch.from_numpy(heat[None, ...]).float()  # (1,H,W)
        return x, y
