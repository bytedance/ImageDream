# ImageDream - threestudio
Peng Wang, Yichun Shi

| [Project Page](https://image-dream.github.io/) | [Paper](https://arxiv.org/abs/2308.16512) | [Gallery](https://mv-dream.github.io/gallery_0.html) 


- **For diffusion model and 2D image generation** check ```./extern/ImageDream```

![imagedream-threestudio-teaser](https://github.com/bytedance/imagedream-threestudio/assets/21265012/b2fef804-7f3f-4b3a-a1a9-8b51596deb54)

## Installation

### Install threestudio

**This part is the same as original threestudio. Skip it if you already have installed the environment.**

See [installation.md](docs/installation.md) for additional information, including installation via Docker.

- You must have an NVIDIA graphics card with at least 20GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- (Optional, Recommended) Create a virtual environment:

```sh
python3 -m virtualenv venv
. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```

- Install `PyTorch >= 1.12`. We have tested on `torch1.12.1+cu113` and `torch2.0.0+cu118`, but other versions should also work fine.

```sh
# torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

### Install imagedream
imagedream multi-view diffusion model is provided in a different codebase. Install it by:

```sh
git clone https://github.com/bytedance/imagedream extern/imagedream
pip install -e extern/imagedream 
```


## Quickstart

In the paper, we use the configuration with soft-shading. It would need an A100 GPU in most cases to compute normal:
```sh
# imagedream with shading (used in paper)
python launch.py --config configs/imagedream-sd21-shading.yaml --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse"
```

## Tips
- Try to use object image with 0 elevation to obtain best result. Place the object at the center of image. 
- **Preview**.  May adopt [ImageDream](https://github.com/bytedance/imagedream) to test if the model can really understand the image and text before using it for 3D generation.
- **Other** Try to refer to [imagedream-threestudio]() for more tips in optimization configuration.


## Credits
This code is built on the [threestudio-project](https://github.com/threestudio-project/threestudio). Thanks to the maintainers for their contribution to the community!


## Cite
```
@article{pengwang2023ImageDream,
  author = {Wang, Peng and Shi, Yichun},
  title = {ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation},
  journal = {arXiv: to-be-update},
  year = {2023},
}
```
