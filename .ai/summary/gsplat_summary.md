# gsplat Project Summary

> **Purpose**: Quick onboarding for AI agents. `gsplat/` is an independent CUDA-accelerated 3D Gaussian rasterization **library** (Nerfstudio / UC Berkeley et al.), vendored inside FastGS repo but **not used by FastGS training**.

- Docs: https://docs.gsplat.studio/
- Paper: gsplat library (JMLR 2025, arXiv:2409.06765)
- Install: `pip install gsplat` or `pip install -e gsplat/`

## 1. What gsplat Is

A **modular Python + CUDA library** for Gaussian splatting — faster and more memory-efficient than official 3DGS, with integrations from many follow-up papers:

| Feature | Source / Notes |
|---------|----------------|
| Core 3DGS rasterization | Faster than Inria official |
| 2DGS (2D Gaussians) | Surface-focused splatting |
| 3DGUT | Distortion, rolling shutter, fisheye/f-theta |
| MCMC densification | MCMC-3DGS style strategy |
| Arbitrary batching | Multi-scene × multi-view in one call |
| LiDAR rasterization | Spinning lidar, depth/hit-distance modes |
| AccuTile | Tighter tile-Gaussian intersection |
| HiGS inference path | Experimental low-latency inference rendering |
| PPISP | Bilateral grid appearance compensation |
| NCore v4 | Autonomous driving capture format |
| Dynamic surgical (G-SHARP port) | 4D Gaussians, HexPlane, EndoNeRF |
| Compression | PNG-based Gaussian compression |
| SelectiveAdam | Sparse optimizer for visible Gaussians |

## 2. Repository Layout

```
gsplat/
├── gsplat/                    # Main package
│   ├── __init__.py            # Public API exports
│   ├── rendering.py           # rasterization(), rasterization_2dgs(), configs
│   ├── cuda/                  # Core CUDA kernels + _wrapper.py
│   │   └── csrc/              # Projection, rasterize, SH, MCMC, losses, ...
│   ├── strategy/              # DefaultStrategy, MCMCStrategy
│   ├── optimizers/            # SelectiveAdam
│   ├── losses.py / losses_fused.py
│   ├── compression/           # PngCompression
│   ├── sensors/               # Camera + LiDAR models (pinhole, fisheye, f-theta, lidar)
│   ├── geometry/              # Pose, quaternion ops
│   ├── scene/                 # GaussianScene, inference scene packing
│   ├── stage/                 # Training stage helpers
│   ├── experimental/render/   # HiGS inference path
│   ├── contrib/dynamic/       # 4D GS: HexPlane, DeformNetwork, DynamicStrategy
│   ├── distributed.py         # Multi-GPU helpers
│   └── exporter.py            # export_splats
├── examples/
│   ├── simple_trainer.py      # Main 3DGS training example (COLMAP / NCore)
│   ├── simple_trainer_2dgs.py
│   ├── simple_viewer.py / simple_viewer_3dgut.py
│   ├── dynamic_surgical_trainer.py
│   ├── av_trainer.py            # Autonomous driving
│   ├── datasets/              # colmap.py, ncore.py, endonerf.py
│   └── benchmarks/            # basic.sh, 3dgut/, compression/, ...
├── tests/                     # Extensive pytest suite
├── docs/                      # Sphinx docs + 3dgut.md, batch.md, modules-design.md
└── setup.py / pyproject.toml
```

## 3. Architecture Pattern

Modules follow a layered design (`docs/modules-design.md`):

```
functional/   → Public stateless API (what users import)
kernels/      → Backend dispatch, autograd, CUDA bindings
models/       → Optional nn.Module wrappers
components/   → Stateful non-Module helpers
```

Key sub-packages: `geometry`, `sensors`, `scene`, `stage` each follow this pattern.

## 4. Core API: `rasterization()`

Primary entry in `gsplat.rendering`:

```python
from gsplat import rasterization

renders, alphas, meta = rasterization(
    means, quats, scales, opacities, colors,  # Gaussian params
    viewmats, Ks, width, height,            # Cameras
    # Optional:
    sh_degree=3,
    camera_model="pinhole",  # pinhole | fisheye | ftheta
    with_ut=False, with_eval3d=False,       # 3DGUT
    render_mode="RGB",       # RGB | D | ED | d | RGB+D | ...
    rasterize_mode="classic", # classic | antialiased
    # Distortion coeffs, rolling shutter, batch dims supported
)
```

**Batching** (v1.5.3+): arbitrary `(N_scenes, N_views, ...)` — see `docs/batch.md`.

Wrappers: `rasterization_inria_wrapper`, `rasterization_2dgs`, `rasterization_2dgs_inria_wrapper`.

## 5. Training Flow (`examples/simple_trainer.py`)

Typical training loop (simplified):

```
1. Load dataset (COLMAP via datasets/colmap.py or NCore)
2. Init Gaussians from point cloud (knn scales)
3. Strategy = DefaultStrategy() or MCMCStrategy()
4. Each step:
   a. rasterization() → RGB
   b. Loss: L1 + SSIM + optional depth/opacity/scale regs
   c. strategy.step_pre_backward() / step_post_backward()
   d. SelectiveAdam or torch.optim
5. Checkpoint as .pt, optional PNG compression
```

Config via **tyro** dataclass `Config` — data_dir, camera_model, strategy, 3DGUT flags, etc.

### Densification Strategies (`gsplat/strategy/`)

| Strategy | Use case |
|----------|----------|
| `DefaultStrategy` | Standard clone/split/prune (vanilla 3DGS style) |
| `MCMCStrategy` | MCMC-3DGS; **required for 3DGUT** |
| `DynamicStrategy` | 4D dynamic scenes (contrib) |

## 6. Major Feature Modules

### 6.1 3DGUT (`docs/3dgut.md`)

Nonlinear camera models without pre-undistortion:

```bash
python examples/simple_trainer.py mcmc --with_ut --with_eval3d ...
```

API flags: `with_ut=True`, `with_eval3d=True`, distortion coeffs, `camera_model`, `rolling_shutter`.

### 6.2 2DGS

`rasterization_2dgs()` — 2D Gaussian surfels for surface reconstruction.

### 6.3 LiDAR (`pip install "gsplat[lidar]"`)

- `isect_tiles_lidar`, `rasterize_to_pixels_eval3d`
- Spinning lidar models in `sensors/models/lidars/`
- Render modes: depth, hit distance, intensity

### 6.4 Inference Rendering (HiGS) — Experimental

Low-latency inference-only path (no training gradients):

```python
from gsplat.experimental import render_scene, GaussianInferenceScene
```

Macro-tile fused rasterization, fp16 scene packing. Benchmark: `examples/benchmarks/gaussian_render_inference_scene/`.

### 6.5 Dynamic Surgical (`contrib/dynamic/`)

G-SHARP v0.2 port for EndoNeRF:

- `HexPlaneField` + `DeformNetwork` + `DynamicStrategy`
- Trainer: `examples/dynamic_surgical_trainer.py`
- Depth supervision, tool masking, two-stage coarse→fine schedule

### 6.6 Sensors (`gsplat/sensors/`)

Unified camera/lidar modeling:

- Pinhole (OpenCV distortion), fisheye, f-theta
- External distortion (windshield rigs)
- TorchScript custom ops for deployment

### 6.7 Compression & Export

- `PngCompression` — compress Gaussian attributes to PNG
- `export_splats()` — export to various formats

## 7. CUDA Backend (`gsplat/cuda/`)

Built JIT on first import or at install time. Key ops in `_wrapper.py`:

| Op | Purpose |
|----|---------|
| `fully_fused_projection` | Project Gaussians to 2D |
| `fully_fused_projection_with_ut` | 3DGUT projection |
| `fully_fused_projection_2dgs` | 2DGS projection |
| `isect_tiles` / `isect_offset_encode` | Tile intersection (AccuTile) |
| `rasterize_to_pixels` | Forward/backward rasterization |
| `spherical_harmonics` | SH evaluation |
| `quat_scale_to_covar_preci` | Covariance from quat+scale |
| MCMC perturb (`inject_noise`) | MCMC strategy acceleration |
| `rasterize_num_contributing_gaussians` | Analysis / debugging |

Capability flags: `has_3dgs`, `has_2dgs`, `has_3dgut`, `has_adam`, `has_losses`, `has_reloc`.

## 8. Examples Index

| Script | Purpose |
|--------|---------|
| `simple_trainer.py` | Standard 3DGS on COLMAP / NCore |
| `simple_trainer_2dgs.py` | 2DGS training |
| `simple_viewer.py` | Interactive viewer (viser) |
| `simple_viewer_3dgut.py` | 3DGUT viewer with distortion |
| `dynamic_surgical_trainer.py` | EndoNeRF surgical scenes |
| `av_trainer.py` | Autonomous driving (NCore) |
| `image_fitting.py` | 2D image → 3D Gaussians demo |
| `benchmarks/basic.sh` | MipNeRF360 reproduction |

## 9. Installation & Build

```bash
cd gsplat
pip install -e .                    # build CUDA at install
# or
pip install gsplat                  # JIT build on first run
# Pre-built wheels: docs.gsplat.studio/whl
```

Requirements: PyTorch + CUDA matching toolchain. Windows: `docs/INSTALL_WIN.md`.

## 10. Testing

```bash
cd gsplat
pytest tests/                       # large suite
bash examples/benchmarks/basic.sh   # eval reproduction
```

Tests organized by module: `test_rasterization.py`, `sensors/`, `scene/`, `experimental/`, `contrib/`, etc.

## 11. FastGS vs gsplat

| | FastGS | gsplat |
|--|--------|--------|
| Type | Training framework (paper repo) | General library |
| Rasterizer | `diff-gaussian-rasterization_fastgs` | Own CUDA in `gsplat/cuda` |
| Densify | Custom multi-view FastGS logic | Strategy classes (Default/MCMC/Dynamic) |
| Training entry | `train.py` | `examples/simple_trainer.py` |
| Output | `point_cloud.ply` | `.pt` checkpoint + optional PLY export |
| Integration | None between them in this repo |

**If extending FastGS**: modify FastGS code paths above.
**If using gsplat features**: work inside `gsplat/` with its API and examples.
**Potential future work**: port FastGS multi-view densification to gsplat `Strategy` interface.

## 12. Common Development Tasks

| Task | Where |
|------|-------|
| New rasterization feature | `gsplat/cuda/csrc/`, `_wrapper.py`, `rendering.py` |
| New densification strategy | `gsplat/strategy/` |
| New camera model | `gsplat/sensors/` |
| New training example | `gsplat/examples/` |
| Add loss | `gsplat/losses.py` or `losses_fused.py` |
| Inference optimization | `gsplat/experimental/render/` |
| Dynamic / 4D GS | `gsplat/contrib/dynamic/` |
| Docs | `gsplat/docs/source/` (Sphinx) |

## 13. Recent Unreleased Features (main branch)

- HiGS inference rendering (May 2026)
- Native CUDA MCMC perturb (May 2026)
- AccuTile ellipse intersection (Apr 2026)
- NCore v4 capture (Apr 2026)
- LiDAR rasterization (Mar 2026)
- TorchScript deployment ops (Mar 2026)
- 3DGUT external distortion, per-ray gradients (Mar 2026)
- PPISP appearance (Jan 2026)
