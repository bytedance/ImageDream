
# # test pixel version
# python scripts/demo.py  \
#     --image "./assets/astronaut.png" \
#     --text "an astronaut riding a horse" \
#     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml" \
#     --ckpt_path "./release_models/sd-v2.1-base-4view-ipmv.pt" \
#     --mode "pixel"
#     --num_frames 5

# test local version
python scripts/demo.py  \
    --image "./assets/astronaut.png" \
    --text "an astronaut riding a horse" \
    --config_path "./imagedream/configs/sd_v2_base_ipmv_local.yaml" \
    --ckpt_path "./release_models/sd-v2.1-base-4view-ipmv-local.pt" \
    --mode "local" \
    --num_frames 4
