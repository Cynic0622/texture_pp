# Texture++: Elevating 3D Asset Texture Resolution with a Region-Aware Diffusion Model

**SIGGRAPH Asia 2026**

Shuaiwei Wang, Shi Li, Jieting Xu, Yuchi Huo, Qi Wang, Wenting Zheng, Rengan Xie

**[Project Page](https://cynic0622.github.io/Texture-pp/)** · **[Paper](https://arxiv.org/abs/2607.21504)** · **[Hugging Face Model](https://huggingface.co/wshw/texture_pp)**

<p align="center">
  <img src="assets/pipeline.png" alt="Texture++ pipeline overview" width="100%">
</p>

We present Texture++, a novel framework for texture super-resolution, which enhances the low-resolution textures of assets to produce high-resolution, high-quality results. Specifically, we reformulate the task of super-resolution in UV space into performing it across multiple rendered views and merging the outputs. Firstly, to achieve more complete and continuous textures in the view space, we propose an adaptive view selection strategy to integrate textures dispersed across UV texture patches. Furthermore, we introduce a quadtree-based texture region organization method for combining super-resolved textures from different viewpoints, providing masks to distinguish regions that require improvement. Finally, we design a diffusion-based super-resolution model that enhances the texture resolution for specified masked regions, seamlessly integrating with surrounding regions.

## Usage

```bash
git clone <repo-url>
cd texture_pp
conda create -n texture_pp python=3.10
conda activate texture_pp
pip install -r requirement.txt

# single model (run from the repo root)
python src/main.py -m <model.obj> [-t <tex.png>]

# directory input (the directory must contain <dirname>.obj and <dirname>.png)
python src/main.py -m <model_dir>
```

## Model Files and Weight Paths

Placed under `models/` in the repo root by default (overridable via CLI arguments) available at [Hugging Face](https://huggingface.co/wshw/texture_pp):

| Weight | Default path |
|--------|-------------|
| TextureSRModel | `models/texture_sr_model/` |

## Directory Structure

```
texture_pp/
├── src/
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

## Citation
If you find Texture++ useful, please cite:
```bibtex
@article{wang2026texturepp,
      title={Texture++: Elevating 3D Asset Texture Resolution with a Region-Aware Diffusion Model},
      author={Shuaiwei Wang and Shi Li and Jieting Xu and Yuchi Huo and Qi Wang and Wenting Zheng and Rengan Xie},
      year={2026},
      eprint={2607.21504},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.21504},
}
```


## License

This project is licensed under the MIT License.

