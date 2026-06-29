# FastGS Project Summary

> **Purpose**: Quick onboarding for AI agents. FastGS is a CVPR 2026 acceleration framework for 3D Gaussian Splatting (3DGS) training — ~100s on MipNeRF360 vs minutes for vanilla 3DGS, with comparable quality.

## 1. What FastGS Is

| Aspect | Detail |
|--------|--------|
| Paper | FastGS: Training 3D Gaussian Splatting in 100 Seconds (arXiv:2511.04283) |
| Base | Fork of vanilla [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) |
| Influences | Taming-3DGS, Speedy-Splat, Abs-GS |
| Output format | Standard 3DGS PLY — compatible with SIBR viewer / Supersplat |
| Relation to `gsplat/` | **Separate project** in same repo; FastGS does **not** use gsplat for training |

### Core Value Proposition

1. **Multi-view consistent densification** — scores Gaussians across sampled views before clone/split (vs single-view gradient only in vanilla 3DGS).
2. **Modified CUDA rasterizer** (`diff-gaussian-rasterization_fastgs`) — compact tile box (`mult`), metric-map accumulation for per-Gaussian scoring.
3. **Strict Gaussian budget control** — scene-bound init filter, distance-based prune, staged final prune after 15k iters.
4. **Abs-GS style split** — uses absolute gradient (`grad_abs_thresh`) for split, vanilla gradient (`grad_thresh`) for clone.

## 2. Repository Layout

```
FastGS/
├── train.py              # Main training entry
├── render.py             # Inference / export images
├── metrics.py            # PSNR / SSIM / LPIPS evaluation
├── convert.py            # COLMAP data prep
├── full_eval.py          # Batch evaluation
├── train_base.sh         # FastGS standard config (densify_interval=500)
├── train_big.sh          # FastGS-Big (densify_interval=100, more Gaussians)
├── arguments/            # CLI param groups (Model/Pipeline/Optimization)
├── scene/
│   ├── __init__.py       # Scene loader (COLMAP / Blender)
│   ├── gaussian_model.py # Gaussian params + densify/prune logic
│   ├── cameras.py        # Camera class
│   └── dataset_readers.py
├── gaussian_renderer/
│   └── __init__.py       # render_fastgs() → CUDA rasterizer
├── utils/
│   ├── fast_utils.py     # compute_gaussian_score_fastgs, sampling_cameras
│   ├── loss_utils.py, image_utils.py, sh_utils.py, ...
├── submodules/
│   ├── diff-gaussian-rasterization_fastgs/  # Custom rasterizer (KEY)
│   ├── fused-ssim/                          # Fast SSIM for loss
│   └── simple-knn/                          # KNN for init scale
├── lpipsPyTorch/         # LPIPS for eval
└── gsplat/               # Independent gsplat library (see gsplat_summary.md)
```

## 3. Training Pipeline (`train.py`)

### High-Level Flow

```
1. Scene(dataset) → load COLMAP/Blender, init Gaussians from SfM point cloud
2. For each iteration:
   a. Random camera → render_fastgs() → L1 + fused SSIM loss → backward
   b. Every 1000 iters: oneupSHdegree()
   c. If iter < densify_until_iter (15k):
      - Accumulate 2D position gradients (vanilla + abs)
      - Every densification_interval iters (after densify_from_iter=500):
        → sampling_cameras(10 random views)
        → compute_gaussian_score_fastgs()  # multi-view metric
        → densify_and_prune_fastgs()
      - Every opacity_reset_interval (3k): reset_opacity()
   d. If iter % 3000 == 0 and 15k < iter < 30k:
      → final_prune_fastgs()  # aggressive late-stage prune
   e. optimizer step (default Adam or sparse_adam)
3. Save PLY at save_iterations
```

### Loss

```python
loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - fused_ssim)
# default: lambda_dssim = 0.2
```

## 4. Key Algorithms (FastGS-specific)

### 4.1 Multi-View Gaussian Scoring (`utils/fast_utils.py`)

`compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY=False)`:

1. For each of 10 sampled cameras:
   - Render image, compute per-pixel normalized L1 loss map.
   - `metric_map = (l1_loss_norm > loss_thresh)` — high-error pixels.
   - Re-render with `get_flag=True` + `metric_map` → rasterizer returns `accum_metric_counts` per Gaussian (how many flagged pixels it covers).
2. `importance_score` = floor-averaged view counts (for densify filter).
3. `pruning_score` = normalized photometric-weighted score (for prune priority).

### 4.2 Densification (`scene/gaussian_model.py`)

`densify_and_prune_fastgs()` steps:

1. **Gradient qualifiers**: `grad_thresh` (clone), `grad_abs_thresh` (split, Abs-GS style).
2. **Size qualifiers**: `dense * extent` — small → clone, large → split.
3. **Multi-view filter**: `metric_mask = importance_score > 5` — only densify Gaussians consistently bad across views.
4. **Prune**: low opacity, large screen radius (>20 after opacity reset), large world scale, far from scene center (`scene_prune_dist_mult * extent`).
5. **Budgeted prune**: sample 50% of prune candidates by inverse pruning_score.

`final_prune_fastgs()`: after 15k — remove low opacity OR `pruning_score > 0.9`.

### 4.3 Scene Bounds (`scene/__init__.py`)

- Init: filter SfM points beyond `scene_init_dist_mult * cameras_extent` (default 2.0).
- Densify prune: remove Gaussians beyond `scene_prune_dist_mult * extent` (default 1.3).

### 4.4 Rasterizer (`gaussian_renderer/__init__.py`)

`render_fastgs()` uses `diff_gaussian_rasterization_fastgs.GaussianRasterizer`:

- `mult`: compact box multiplier → controls tile count per splat (Speedy-Splat style, default 0.5).
- `get_flag` + `metric_map`: second-pass render to accumulate per-Gaussian metric counts.
- Separate SH DC / rest (`separate_sh=True` in pipeline).
- `screenspace_points` shape `(N, 4)` — grad on [:2] for vanilla, [2:] for abs gradient.

## 5. Important CLI Parameters

| Param | Default | Role |
|-------|---------|------|
| `loss_thresh` | 0.1 | Pixel L1 threshold for metric_map |
| `grad_thresh` | 0.0002 | Clone gradient threshold |
| `grad_abs_thresh` | 0.0012 | Split gradient threshold (Abs-GS) |
| `dense` | 0.001 | Size fraction of extent for clone/split split |
| `mult` | 0.5 | Rasterizer tile compactness |
| `densification_interval` | 100 (base.sh: 500) | Densify frequency |
| `densify_until_iter` | 15000 | Stop densification |
| `highfeature_lr` / `lowfeature_lr` | 0.005 / 0.0025 | SH rest / DC learning rates |
| `scene_init_dist_mult` | 2.0 | Init point cloud distance filter |
| `scene_prune_dist_mult` | 1.3 | Runtime distance prune |
| `optimizer_type` | default | `default` (Adam) or `sparse_adam` |

Per-scene tuning in `train_base.sh` / `train_big.sh` mainly adjusts `grad_abs_thresh`, `highfeature_lr`, `dense`, `mult`.

## 6. Data Format

- **COLMAP**: `sparse/` + `images/` (or custom via `-i`).
- **Blender NeRF**: `transforms_train.json`.
- **Eval split**: `--eval` for MipNeRF360-style holdout.
- **Resolution**: auto downscale if width > 1600px; force full res with `-r 1`.

```
datasets/mipnerf360/bicycle/
datasets/tanksandtemples/truck/
datasets/db/playroom/
```

## 7. Environment

- Conda: `environment.yml` → `conda activate fastgs`
- CUDA 11.x for extension build (`diff-gaussian-rasterization_fastgs`)
- Recommended: RTX 4090, 24GB VRAM
- Clone with `--recursive` for submodules

## 8. Extension Branches (separate git branches, not main)

| Branch | Base method | Task |
|--------|-------------|------|
| fast-d3dgs | Deformable-3D-Gaussians | Dynamic scenes |
| fast-dropgaussian | DropGaussian | Sparse-view |
| fast-pgsr | PGSR | Surface reconstruction |

Main branch: static COLMAP scenes only.

## 9. Common Development Tasks

| Task | Where to look |
|------|---------------|
| Change densify logic | `scene/gaussian_model.py` |
| Change multi-view scoring | `utils/fast_utils.py` |
| Rasterizer / tile behavior | `submodules/diff-gaussian-rasterization_fastgs/` |
| Add dataset format | `scene/dataset_readers.py` |
| New CLI args | `arguments/__init__.py` (OptimizationParams) |
| Render / export | `render.py` (uses same `render_fastgs`, pass `--mult`) |

## 10. Key Differences vs Vanilla 3DGS

| | Vanilla 3DGS | FastGS |
|--|--------------|--------|
| Densify signal | Single-view 2D grad | Multi-view photometric consistency |
| Split criterion | grad only | abs grad (Abs-GS) |
| Rasterizer | diff-gaussian-rasterization | `_fastgs` fork with `mult`, metric_map |
| Late prune | opacity/size only | + aggressive multi-view prune 15k–30k |
| Init | all SfM points | distance-filtered |
| Training time | ~5–30 min | ~100s (base config) |
