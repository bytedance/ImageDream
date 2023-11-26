import sys

from dataclasses import dataclass, field

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms as T
from imagedream.ldm.util import add_random_background
from imagedream.camera_utils import convert_opengl_to_blender, normalize_camera
from imagedream.model_zoo import build_model

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("multiview-diffusion-guidance")
class MultiviewDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = (
            "sd-v2.1-base-4view"  # check imagedream.model_zoo.PRETRAINED_MODELS
        )
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        config_path: Optional[
            str
        ] = None # path to local config (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5
        ip_mode: str = None

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model = build_model(
            self.cfg.model_name, 
            config_path=self.cfg.config_path,
            ckpt_path=self.cfg.ckpt_path)
            
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.to(self.device)

        threestudio.info(f"Loaded Multiview Diffusion!")

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32] Latent space image


    def append_extra_view(self, latent_input, t_expand, context, ip=None):
        """
        Args: 
            latent_input: [BZ, C, H, W]
            context: dict that contain text, camera, image embeddings
            ip: the input image
        """
        # append another view in the image
        # append in latent input, camera, context ip
        image_transform = T.Compose(
            [
                T.Resize((self.cfg.image_size, self.cfg.image_size)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        bs, c, h, w = latent_input.shape
        real_batch_size = bs // self.cfg.n_view
        latent_input = latent_input.reshape(real_batch_size, -1, c, h, w)
        zero_tensor = torch.zeros(real_batch_size, 1, c, h, w).to(latent_input)
        latent_input = torch.cat([latent_input, zero_tensor], dim=1)
        latent_input = latent_input.reshape(-1, c, h, w)
        
        # make time expand here
        t_expand = torch.cat([t_expand, t_expand[-1:].repeat(real_batch_size)])
        
        # repeat 
        for key in ["context", "ip"]:
            embedding = context[key] # repeat for last dim features
            features = []
            for feature in embedding.chunk(real_batch_size):
                features.append(torch.cat([feature, feature[-1].unsqueeze(0)], dim=0))
            context[key] = torch.cat(features, dim=0)
        
        # set 0
        for key in ["camera"]:
            embedding = context[key]
            features = []
            for feature in embedding.chunk(real_batch_size):
                zero_tensor = torch.zeros_like(feature[0]).unsqueeze(0).to(feature)
                features.append(torch.cat([feature, zero_tensor], dim=0))
            context[key] = torch.cat(features, dim=0)
        
        if ip:
            ip = image_transform(ip).to(latent_input)
            ip_img = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(ip[None, :, :, :]))
            ip_pos_num = real_batch_size // 2
            ip_img = ip_img.repeat(ip_pos_num, 1, 1, 1)
            context["ip_img"] = torch.cat([
                ip_img, 
                torch.zeros_like(ip_img)], dim=0) # 2 * (batchsize + 1, c, h, w)
        
        return latent_input, t_expand, context
    
    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        extra_view = self.cfg.ip_mode == "pixel"    
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            
        ip = None
        if prompt_utils.image is not None:
            ip = prompt_utils.image
            bg_color = kwargs.get("comp_rgb_bg")
            bg_color = bg_color.mean().detach().cpu().numpy() * 255 
            ip = add_random_background(ip, bg_color)
            image_embeddings = self.model.get_learned_image_conditioning(ip)
            un_image_embeddings = \
                torch.zeros_like(image_embeddings).to(image_embeddings)
                    
        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (
                    F.interpolate(
                        rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                    )
                    * 2
                    - 1
                )
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2, 1).to(text_embeddings)
                num_frames = self.cfg.n_view + 1 if extra_view else self.cfg.n_view
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": num_frames, # number of frames
                }
            else:
                context = {"context": text_embeddings}
                
            if prompt_utils.image is not None:
                context["ip"] = torch.cat([
                    image_embeddings.repeat(batch_size, 1, 1), 
                    un_image_embeddings.repeat(batch_size, 1, 1)], dim=0).to(text_embeddings)
            
            if extra_view:
                latent_model_input, t_expand, context = \
                    self.append_extra_view(latent_model_input, t_expand, context, ip=ip)     
                    
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(
            2
        )  # Note: flipped compared to stable-dreamfusion
        
        if extra_view:
            _, c, h, w = noise_pred_text.shape
            def remove_extra_view(embedding):
                embedding = embedding.reshape(-1, (self.cfg.n_view + 1), c, h, w)
                embedding = embedding[:, :-1, :, :, :].reshape(-1, c, h, w)
                return embedding
            
            noise_pred_text, noise_pred_uncond = \
                remove_extra_view(noise_pred_text), \
                remove_extra_view(noise_pred_uncond)
                
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(
                latents_noisy, t, noise_pred
            )

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(
                    latents_noisy, t, noise_pred_text
                )
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                    -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                )
                latents_recon_reshape = latents_recon.view(
                    -1, self.cfg.n_view, *latents_recon.shape[1:]
                )
                factor = (
                    latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8
                ) / (latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

                latents_recon_adjust = latents_recon.clone() * factor.squeeze(
                    1
                ).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = (
                    self.cfg.recon_std_rescale * latents_recon_adjust
                    + (1 - self.cfg.recon_std_rescale) * latents_recon
                )

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = (
                0.5
                * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
                / latents.shape[0]
            )
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = 1 - self.alphas_cumprod[t]
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
