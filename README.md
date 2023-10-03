# MVDream - threestudio
Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, Xiao Yang

| [Project Page](https://mv-dream.github.io/) | [Paper](https://arxiv.org/abs/2308.16512) |


- **This code is forked from [threestudio](https://github.com/threestudio-project/threestudio) for SDS and 3D Generation using MVDream.**
- **For diffusion model and 2D image generation, check original [MVDream](https://github.com/bytedance/MVDream) repo.**

![mvdream-threestudio-teaser](https://github.com/bytedance/MVDream-threestudio/assets/21265012/0596fd14-9bbe-4b10-9ef8-fccf83a1412e)

## Installation

### Install threestudio

**This part is the same as original threestudio. Skip it if you already have installed the environment.**

See [installation.md](docs/installation.md) for additional information, including installation via Docker.

The following steps have been tested on Ubuntu20.04.

- You must have an NVIDIA graphics card with at least 6GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
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

- (Optional, Recommended) The best-performing models in threestudio use the newly-released T2I model [DeepFloyd IF](https://github.com/deep-floyd/IF), which currently requires signing a license agreement. If you would like to use these models, you need to [accept the license on the model card of DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), and login into the Hugging Face hub in the terminal by `huggingface-cli login`.

- For contributors, see [here](https://github.com/threestudio-project/threestudio#contributing-to-threestudio).

### Install MVDream
MVDream multi-view diffusion model is provided in a different codebase. Install it by:

```sh
git clone https://github.com/bytedance/MVDream extern/MVDream
pip install -e extern/MVDream 
```


## Quickstart

We currently provide two configurations for MVDream, one without soft-shading and one with it. The one without shading is more effecient in both memory and time. You can run it by

```sh
# MVDream without shading (memory efficient)
python launch.py --config configs/mvdream-sd21.yaml --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse"
```

In the paper, we use the configuration with soft-shading. It would need an A100 GPU in most cases to compute normal:
```sh
# MVDream without shading (used in paper)
python launch.py --config configs/mvdream-sd21-shading.yaml --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse"
```

### Resume from checkpoints

If you want to resume from a checkpoint, do:

```sh
# resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
# if the training has completed, you can still continue training for a longer time by setting trainer.max_steps
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt trainer.max_steps=20000
# you can also perform testing using resumed checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --test --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
# note that the above commands use parsed configuration files from previous trials
# which will continue using the same trial directory
# if you want to save to a new trial directory, replace parsed.yaml with raw.yaml in the command

# only load weights from saved checkpoint but dont resume training (i.e. dont load optimizer state):
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 system.weights=path/to/trial/dir/ckpts/last.ckpt
```

## Tips

- **Rescale Factor**. We introducte rescale adjustment from [Shanchuan et al.](https://arxiv.org/abs/2305.08891) to alleviate the texture over-saturation from large CFG guidance. However, in some cases, we find it to cause floating noises in the generated scene and consequently OOM issue. Therefore we reduce the rescale factor from 0.7 in original paper to 0.5. However, if you still encoder such problem, please try to further reduce `system.guidance.recon_std_rescale=0.3`.

## Credits

This code is built on the [threestudio-project](https://github.com/threestudio-project/threestudio). Thanks to the maintainers for their contribution to the community!

## Citing

If you find MVDream helpful, please consider citing:

```
@article{shi2023MVDream,
  author = {Shi, Yichun and Wang, Peng and Ye, Jianglong and Mai, Long and Li, Kejie and Yang, Xiao},
  title = {MVDream: Multi-view Diffusion for 3D Generation},
  journal = {arXiv:2308.16512},
  year = {2023},
}
```
