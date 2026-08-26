# texture_pp

Multi-view rendering + mask-conditioned diffusion model for 3D texture super-resolution.

## Overview

Takes an OBJ mesh with a low-resolution UV texture, renders it from multiple
views, super-resolves each view with a mask-guided local SR model, and
gathers the results back into texture space per-texel to produce a 4x
super-resolved texture.

## Usage

```bash
pip install -r requirement.txt

# single model
python texture_pp/main.py -m <model.obj> [-t <tex.png>]

# directory input (the directory must contain <dirname>.obj and <dirname>.png)
python texture_pp/main.py -m <model_dir>
```

## Model Files and Weight Paths

Placed under `models/` in the repo root by default (overridable via CLI arguments):

| Weight | Default path |
|--------|-------------|
| SD 2.1 base | `models/stable-diffusion-2-1-base/` |
| TextureSRModel checkpoint | `models/texture_sr.pkl` |

## Directory Structure

```
texture_pp/
├── texture_pp/
│   ├── main.py              CLI entry
│   ├── pipeline.py          inference loop (super_resolve_texture)
│   ├── config.py            Config
│   ├── io_utils/            OBJ loading, texture loading, image saving
│   ├── geometry/            UV-island / seam view selection
│   ├── rendering/           projection matrix, rasterisation, texture-space LUTs
│   ├── quality/             view quality Q_view, quadtree mask refinement
│   ├── projection/          per-texel gather back-projection
│   └── sr/
│       ├── local.py         LocalSRModel wrapper
│       └── model/           TextureSRModel and SD network definitions
└── requirement.txt
```
