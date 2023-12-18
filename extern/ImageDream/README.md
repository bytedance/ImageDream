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

To use ImageDream as a python module, you can install it by `pip install -e .` or:
```bash
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream
```

## Image-to-Multi-View
Clone the modelcard on the [Huggingface ImageDream Model Page](https://huggingface.co/Peng-Wang/ImageDream/) under ```./release_models/```

Replace the object in the center of RGBA image and a short description of the image is necessary to obtain good results. For image only case, one may run a simple caption model such as [Llava](https://llava.hliu.cc/) or [BLIP2](https://huggingface.co/spaces/Salesforce/BLIP2), which may get similar results. This also applies for 3D SDS.


``` bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py  \
    --image "./assets/astronaut.png" \
    --text "an astronaut riding a horse" \
    --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml" \
    --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt" \
    --mode "pixel" \
    --num_frames 5
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
