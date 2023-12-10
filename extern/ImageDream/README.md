# ImageDream Diffusion
Peng Wang, Yichun Shi

| [Project Page](https://image-dream.github.io/) | [Paper](https://arxiv.org/abs/2312.02201) | [HuggingFace Demo]() |

## 
- **This repo inherit content from repos of [LDM](), [MVDream]() and some adaptor module from [IP-Adaptor]()**
- **It only includes the diffusion model and 2D image generation.For 3D Generation, please check [Here](https://github.com/bytedance/ImageDream).**


## Installation
Setup environment as in [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion) for this repo. You can set up the environment by installing the given requirements
``` bash
pip install -r requirements.txt
```

## Model Card
Our models are provided on the [Huggingface ImageDream Model Page](https://huggingface.co/Peng-Wang/ImageDream/) with the OpenRAIL license.
We use the SD-2.1-base model in our experiments.  Note that you don't have to manually download the checkpoints for the following scripts.


## Image-to-Multi-View
Replace the object in the center of RGBA image and a short description of the image is necessary to obtain good results. For image only case, one may run a simple caption model such as [Llava](https://llava.hliu.cc/) or [BLIP2](https://huggingface.co/spaces/Salesforce/BLIP2), which may get similar results. This also applies for 3D SDS.


``` bash
python scripts/imagedream.py  \
  --image "./assets/astronaut.png" \
  --text "an astronaut riding a horse"
```

Tips
- The model is trained with same elevation between the input image prompt and synthesized views. Therefore, may adjust the camera elevation in ```get_camera()``` for better results. In paper, we adopt a unified elevation with 5 degree. This also applied for threestudio fusion for a better results.


## Acknowledgement
This repository is heavily based on [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base). We would like to thank the authors of these work for publicly releasing their code.

## Citation
If you find ImageDream helpful, please consider citing:

``` bibtex
@article{wang2023imagedream,
  title={ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation},
  author={Wang, Peng and Shi, Yichun},
  journal={arXiv preprint arXiv:2312.02201},
  year={2023}
}
```
