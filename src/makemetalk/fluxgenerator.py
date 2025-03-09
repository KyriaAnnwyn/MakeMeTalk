import torch
from einops import rearrange
from PIL import Image
import cv2
import numpy as np
from openai_appearance import get_description
from dotenv import load_dotenv
import re
from typing import Optional
from collections import Counter
import json
import os
import time

load_dotenv(override=True)

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long


def get_models(name: str, device: torch.device, offload: bool, fp8: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    if fp8:
        model = load_flow_model_quintized(name, device="cpu" if offload else device)
    else:
        model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool, aggressive_offload: bool, args):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            fp8=args.fp8,
        )
        self.pulid_model = PuLIDPipeline(self.model, device="cpu" if self.offload else device, weight_dtype=torch.bfloat16,
                                         onnx_provider=args.onnx_provider)
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(args.pretrained_model, version=args.version)

    @torch.inference_mode()
    def generate_image(
            self,
            prompt,
            width = 1024,
            height = 1024,
            num_steps = 30,
            start_step = 0,
            guidance = 7,
            seed = -1,
            id_image=None,
            id_weight=1.0,
            neg_prompt="",
            true_cfg=1.0,
            timestep_to_start_cfg=1,
            max_sequence_length=128,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        # offload TEs to CPU, load processor models and id encoder to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        if id_image is not None:
            id_embeddings = []
            uncond_id_embeddings = []
            for idimg in id_image:
                idimg = resize_numpy_image_long(idimg, 1024)
                id_emb, uncond_id_emb = self.pulid_model.get_id_embedding(idimg, cal_uncond=use_true_cfg)
                id_embeddings.append(id_emb)
                if uncond_id_emb:
                    uncond_id_embeddings.append(uncond_id_emb)
            
            #print(f"uncond_id_embeddings = {uncond_id_embeddings}")
            id_embeddings = torch.mean(torch.stack(id_embeddings), 0) #np.mean(id_embeddings)
            if uncond_id_embeddings:
                uncond_id_embeddings = torch.mean(torch.stack(uncond_id_embeddings), 0)
            #print(f"id_embeddings = {id_embeddings}\n uncond_id_embeddings = {uncond_id_embeddings}")
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # offload processor models and id encoder to CPU, load dit model to gpu
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
            start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img, str(opts.seed), self.pulid_model.debug_img_list

    @torch.inference_mode()
    def generate_by_embeddings(
        self,
        prompt,
            width = 1024,
            height = 1024,
            num_steps = 30,
            start_step = 2,
            guidance = 7,
            seed = -1,
            id_embeddings=None,
            uncond_id_embeddings=None,
            id_weight=1.0,
            neg_prompt="",
            true_cfg=1.0,
            timestep_to_start_cfg=1,
            max_sequence_length=128,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # prepare input
        #print(f"Height = {height}, and in opts {opts.height}")
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        # offload TEs to CPU, load processor models and id encoder to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        # offload processor models and id encoder to CPU, load dit model to gpu
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
            start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img, str(opts.seed), self.pulid_model.debug_img_list

    @torch.inference_mode()
    def generate_embeddings(
        self,
        id_image,
        true_cfg=1.0,
    ):
        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        if id_image is not None:
            id_embeddings = []
            uncond_id_embeddings = []
            genders = []
            for idimg in id_image:
                idimg = resize_numpy_image_long(idimg, 1024)
                id_emb, uncond_id_emb, id_gender = self.pulid_model.get_id_embedding(idimg, cal_uncond=use_true_cfg)
                id_embeddings.append(id_emb)
                genders.append(id_gender)
                if uncond_id_emb:
                    uncond_id_embeddings.append(uncond_id_emb)
            
            id_embeddings = torch.mean(torch.stack(id_embeddings), 0) 
            if uncond_id_embeddings:
                uncond_id_embeddings = torch.mean(torch.stack(uncond_id_embeddings), 0)
            gender = max(genders, key=genders.count)
        else:
            id_embeddings = None
            uncond_id_embeddings = None
            gender = None

        return id_embeddings, uncond_id_embeddings, gender

    def generate_avatar_embedding(self, src_folder: str | list):
        if isinstance(src_folder, str):
            image_basename_list = os.listdir(src_folder)
            image_path_list = sorted([os.path.join(src_folder, basename) for basename in image_basename_list 
                                        if ".jpeg" in basename.lower() 
                                        or ".jpg" in basename.lower() 
                                        or ".png" in basename.lower() 
                                        or ".webp" in basename.lower()
                                        or ".avif" in basename.lower()])

            id_image = [cv2.imread(pth) for pth in  image_path_list]
        else:
            id_image = src_folder
        id_embeddings, uncond_id_embeddings, gender = self.generate_embeddings(id_image=id_image)

        print(f"Detected gender = {gender}")
        gender = "female" if gender == 1 else "male"
        return id_embeddings, uncond_id_embeddings, gender


human_words_v2 = [
    "a person",
    "a girl",
    "a boy",
    "a lady",
    "a guy",
    "male",
    "female",
    "person",
    "girl",
    "boy",
    "lady",
    "human",
    "I",
    "me",
    "persona",
    "guy",
    "myself",
    "self",
]
hum_reg_v2 = re.compile(rf"\b(?:{'|'.join(human_words_v2)})\b", re.IGNORECASE)

def modify_postpic_prompt(prompt: str, gender: Optional[str] = None, appearance: Optional[str] = None) -> str:
    """
    Legacy modificator helper
    """
    if not gender:
        return prompt

    mod = ("a woman " if gender == "female" else "a man ") + appearance
    prompt = re.sub(hum_reg_v2, mod, prompt, count=1)
    if not re.search(r"\bman\b|\bwoman\b", prompt[:30], re.IGNORECASE):
        prompt = mod + " " + prompt

    if "portrait" not in prompt:
        prompt = "portrait photo, " + prompt
    return prompt
