# ImageDream Reconstruction
Peng Wang, Yichun Shi

[Project Page](https://image-dream.github.io/) | [Paper](https://arxiv.org/abs/2312.02201) | [Demo
]()


<!-- ![mvdream-threestudio-teaser](https://github.com/bytedance/MVDream-threestudio/assets/21265012/b2fef804-7f3f-4b3a-a1a9-8b51596deb54) -->


## Installation 

This part is the same as original [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio). Skip it if you already have installed the environment.


## Quickstart
Clone the modelcard on the [Huggingface ImageDream Model Page](https://huggingface.co/Peng-Wang/ImageDream/) under ```./extern/ImageDream/release_models/ImageDream```

In the paper, we use the configuration with soft-shading. It would need an A100 GPU in most cases to compute normal:
```sh
gpu=0
method=imagedream-sd21-shading
name="astronaut"
prompt="an astronaut riding a horse"
image_path="./extern/ImageDream/assets/astronaut.png"
ckpt_path="./extern/ImageDream/release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"
config_path="./extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml"
python3 launch.py \
    --config configs/$method.yaml \
    --train \
    --gpu $gpu \
    name="${method}" \
    tag=${name} \
    system.prompt_processor.prompt="$prompt" \
    system.prompt_processor.image_path="$image_path" \
    system.guidance.ckpt_path="$ckpt_path"  \
    system.guidance.config_path="$config_path" 
```

## Credits
- This code is forked from [threestudio](https://github.com/threestudio-project/threestudio) and [MVDream](https://github.com/bytedance/MVDream-threestudi) for SDS and 3D Generation.
- For diffusion only model, refer to subdir ```./extern/ImageDream/```


## Citing

If you find ImageDream helpful, please consider citing:

``` bibtex
@article{wang2023imagedream,
  title={ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation},
  author={Wang, Peng and Shi, Yichun},
  journal={arXiv preprint arXiv:2312.02201},
  year={2023}
}
```
