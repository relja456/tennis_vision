# import torch
# import torch.nn.functional as F
# from tqdm.auto import tqdm

# from dataset_triplet_frames import DatasetTripletFrames
# from models.tracknet.tracknet import TrackNet


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TrackNet(in_channels=9, num_bins=1).to(device)

# # imbalance handling
# # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0], device=device))

# # optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=19e-4)
# # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
# # scaler = GradScaler("cuda")

# # train_ds = DatasetTripletFrames("/content/dataset", out_size=(360, 640))
# # val_split = int(len(train_ds) * 0.1)
# # train_subset, val_subset = torch.utils.data.random_split(
# #     train_ds, [len(train_ds) - val_split, val_split], generator=torch.Generator().manual_seed(42)
# # )

# # train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
# # val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)


# def train_one_epoch():
#     model.train()
#     total, n = 0.0, 0
#     pbar = tqdm(train_loader, desc="Training", leave=False)

#     optimizer.zero_grad(set_to_none=True)
#     for xb, yb in pbar:
#         xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

#         with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
#             logits = model(xb)
#             if logits.shape[-2:] != yb.shape[-2:]:
#                 logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear", align_corners=False)
#             loss = criterion(logits, yb)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)

#         total += loss.item() * xb.size(0)
#         n += xb.size(0)
#         pbar.set_postfix(loss=f"{loss.item():.4f}")

#     scheduler.step()
#     return total / max(1, n)


# @torch.no_grad()
# def evaluate():
#     model.eval()
#     total, n = 0.0, 0
#     pbar = tqdm(val_loader, desc="Validation", leave=False)
#     for xb, yb in pbar:
#         xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
#         with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
#             logits = model(xb)
#             if logits.shape[-2:] != yb.shape[-2:]:
#                 logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear", align_corners=False)
#             loss = criterion(logits, yb)

#         total += loss.item() * xb.size(0)
#         n += xb.size(0)
#         pbar.set_postfix(loss=f"{loss.item():.4f}")

#     return total / max(1, n)


# def train():
#     best = float("inf")
#     for epoch in range(1, 21):
#         tr = train_one_epoch()
#         vl = evaluate()
#         print(f"epoch {epoch:02d} | train {tr:.4f} | val {vl:.4f}")
#         if vl < best:
#             best = vl
#             torch.save({"model": model.state_dict()}, "/content/tracknet_best.pth")
#             print("  ↳ saved /content/tracknet_best.pth")


# import cv2

# import predictions


# ds = DatasetTripletFrames("datasets/ball_only", out_size=(352, 640))

# clip_idx = len(ds.clip_entries) - 1

# clip_dir = ds.clip_entries[clip_idx]["dir"]
# frames = ds.clip_entries[clip_idx]["frames"]

# for (cx, cy), fname in zip(predictions.predictions, frames):
#     img_path = f"{clip_dir}/{fname}"
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (640, 352))
#     cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
#     cv2.imshow("Prediction", img)
#     if cv2.waitKey(50) & 0xFF == ord("q"):
#         break

# cv2.destroyAllWindows()

