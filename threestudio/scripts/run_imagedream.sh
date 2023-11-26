export PYTHONPATH=$PYTHONPATH:./extern/ImageDream

gpu=0
method=imagedream-sd21-shading
ckpt_path="./extern/ImageDream/release_models/sd-v2.1-base-4view-ipmv.pt"
config_path="./extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml"

name="astronaut"
prompt="an astronaut riding a horse"
image_path="./extern/ImageDream/assets/astronaut.png"

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
