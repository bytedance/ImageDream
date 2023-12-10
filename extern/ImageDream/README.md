# ImageDream Diffusion
Peng Wang, Yichun Shi

| [Project Page](https://image-dream.github.io/) | [Paper](https://arxiv.org/abs/2308.16512) | [HuggingFace Demo]() |

## 
- **This repo inherit content from repos of [LDM](), [MVDream]() and some adaptor module from [IP-Adaptor]()**
- **It only includes the diffusion model and 2D image generation.For 3D Generation, please check [Here](https://github.com/bytedance/ImageDream).**


## Installation
Setup environment as in [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion) for this repo. You can set up the environment by installing the given requirements
``` bash
pip install -r requirements.txt
```

## Model Card
Our models are provided on the [Huggingface Model Page](https://huggingface.co/Peng-Wang/ImageDream/) with the OpenRAIL license.

We use the SD-2.1-base model in our experiments. 
Note that you don't have to manually download the checkpoints for the following scripts.


## Image-to-Multi-View
Notice we will re-place the object in the center of RGBA image. A short description of the image is necessary to obtain good results since we train a model with join modality. For image only case, one may run a simple caption model such as [Llava]() or [BLIP2](), which may get similar results. 

You can simply generate multi-view images of pixel model by running the following command:

``` bash
python scripts/imagedream.py  \
  --image "./assets/astronaut.png" \
  --text "an astronaut riding a horse"
```

We also provide a gradio script to try out with GUI:
``` bash
python scripts/gradio_app.py
```
Tips
- The model is trained with same elevation between the input image prompt and synthesized views. Therefore, may adjust the camera elevation in ```get_camera``` for better results. In paper, we adopt a unified elevation.
- This also applied for threestudio fusion for a better results.


## Acknowledgement
This repository is heavily based on [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base). We would like to thank the authors of these work for publicly releasing their code.

## Citation
``` bibtex
@article{pengwang2023ImageDream,
  author = {Wang, Peng and Shi, Yichun},
  title = {ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation},
  journal = {arXiv-to-be-update},
  year = {2023},
}
```
