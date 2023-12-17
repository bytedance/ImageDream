# ImageDream Reconstruction
Peng Wang, Yichun Shi

[Project Page](https://image-dream.github.io/) | [Paper](https://arxiv.org/abs/2312.02201) | [Demo
]()


<!-- ![mvdream-threestudio-teaser](https://github.com/bytedance/MVDream-threestudio/assets/21265012/b2fef804-7f3f-4b3a-a1a9-8b51596deb54) -->


## Installation 

This part is the same as original [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio). Skip it if you already have installed the environment.


## Quickstart
Clone the modelcard on the [Huggingface ImageDream Model Page](https://huggingface.co/Peng-Wang/ImageDream/) under ```./extern/ImageDream/release_models/```

In the paper, we use the configuration with soft-shading. It would need an A100 GPU in most cases to compute normal:
```sh
image_file="./extern/ImageDream/assets/astronaut.png"
ckpt_file="./extern/ImageDream/release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"
cfg_file="./extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml"

python3 launch.py \
    --config configs/$method.yaml --train --gpu 0 \
    name="imagedream-sd21-shading" tag="astronaut" \
    system.prompt_processor.prompt="an astronaut riding a horse" \
    system.prompt_processor.image_path="${image_file}" \
    system.guidance.ckpt_path="${ckpt_file}" \
    system.guidance.config_path="${cfg_file}"
```


***Check*** ```./threestudio/scripts/run_imagedream.sh``` ***for a bash example.***


## Credits
- This code is forked from [threestudio](https://github.com/threestudio-project/threestudio) and [MVDream](https://github.com/bytedance/MVDream-threestudi) for SDS and 3D Generation.
- For diffusion only model, refer to subdir ```./extern/ImageDream/```

## Tips
1. Place the object in the center and do not make it too large/small in the image.
2. If you have an object cutting image edge, in config, tuning the parameters range of elevation and fov to be a larger range, e.g. ```[0, 30]```, otherwise, you may do image outpainting and follow tips 1.
3. Check the results with ImageDream diffusion model before using it in 3D rendering to save time.


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
