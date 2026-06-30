import numpy as np
import cv2
from pathlib import Path

def save_depth_three_formats(depth, stem, out_dir):
    u16_dir = out_dir / "depth_u16"
    vis_dir = out_dir / "depth_vis"
    npy_dir = out_dir / "depth_npy"
    for d in (u16_dir, vis_dir, npy_dir):
        d.mkdir(parents=True, exist_ok=True)

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    depth_u16 = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    base_name = stem + ".png"

    cv2.imwrite(str(u16_dir / base_name), depth_u16)
    np.save(str(npy_dir / (stem + ".npy")), depth)

    valid_mask = np.isfinite(depth) & (depth > 0)
    if np.any(valid_mask):
        d = depth[valid_mask]
        d_min = np.percentile(d, 2.0)
        d_max = np.percentile(d, 98.0)
        if d_max <= d_min:
            d_max = d_min + 1e-6
        depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
        depth_vis_u8 = (depth_norm * 255.0).astype(np.uint8)
        depth_vis_u8[~valid_mask] = 0
        depth_color = cv2.applyColorMap(depth_vis_u8, cv2.COLORMAP_TURBO)
    else:
        h, w = depth.shape[:2]
        depth_color = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(vis_dir / base_name), depth_color)