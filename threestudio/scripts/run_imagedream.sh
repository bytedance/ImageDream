export PYTHONPATH=$PYTHONPATH:./extern/ImageDream

gpu=0
method=imagedream-sd21-shading
name="astronaut"
prompt="an astronaut riding a horse"
image_path="./extern/ImageDream/assets/astronaut.png"

# for pixel [ImageDream-P]
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

# for local [ImageDream-G]
ckpt_path="./extern/ImageDream/release_models/ImageDream/sd-v2.1-base-4view-ipmv-local.pt"
config_path="./extern/ImageDream/imagedream/configs/sd_v2_base_ipmv_local.yaml"
python3 launch.py \
    --config configs/$method.yaml \
    --train \
    --gpu $gpu \
    name="${method}" \
    tag=${name} \
    system.prompt_processor.prompt="$prompt" \
    system.prompt_processor.image_path="$image_path" \
    system.guidance.ckpt_path="$ckpt_path"  \
    system.guidance.config_path="$config_path" \
    system.guidance.ip_mode="local"