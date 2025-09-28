import time

import torch


@torch.no_grad()
def benchmark_inference_speed(model, data_loader, device=None, warmup=5):
    """
    Measure average iterations (batches) per second during inference.

    Args:
        model: TrackNet
        data_loader: torch DataLoader (only images, no labels needed)
        device: torch.device
        warmup: number of batches to skip for warmup (JIT/cuda init)
    """
    model.eval()
    device = device or model.device

    total_batches, total_time = 0, 0.0

    # warmup
    for i, (xb, _) in enumerate(data_loader):
        xb = xb.to(device)
        _ = model(xb)
        if i + 1 >= warmup:
            break

    # timing loop
    start = time.time()
    for xb, _ in data_loader:
        xb = xb.to(device)
        _ = model(xb)
        total_batches += 1
    total_time = time.time() - start

    iters_per_sec = total_batches / total_time if total_time > 0 else 0
    print(f'⚡ Average speed: {iters_per_sec:.2f} iterations/sec over {total_batches} batches')
    return iters_per_sec
