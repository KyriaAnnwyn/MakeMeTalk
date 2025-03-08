#echomimicimports
import random
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import os

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image

from echomimic_src.models.unet_2d_condition import UNet2DConditionModel
from echomimic_src.models.unet_3d_echo import EchoUNet3DConditionModel
from echomimic_src.models.whisper.audio2feature import load_audio_model
from echomimic_src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline
from echomimic_src.utils.util import save_videos_grid, save_videos_grid_moviepy, crop_and_pad
from echomimic_src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN

def setup_ffmpeg():
    ffmpeg_path = os.getenv('FFMPEG_PATH')
    if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
        try:
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
            if result.returncode == 0:
                ffmpeg_path = result.stdout.strip()
                print(f"FFmpeg is installed at: {ffmpeg_path}")
            else:
                print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
                print("For example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
        except Exception as e:
            pass

    if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
        print("Adding FFMPEG_PATH to PATH")
        os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]

class SpeechAnimator:
    def __init__(self, config):

        self.config = config
        if config.weight_dtype == "fp16":
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        device = config.device
        if device.__contains__("cuda") and not torch.cuda.is_available():
            device = "cpu"

        inference_config_path = config.inference_config
        infer_config = OmegaConf.load(inference_config_path)

        ## vae init
        self.vae = AutoencoderKL.from_pretrained(
            config.pretrained_vae_path,
        ).to(device, dtype=self.weight_dtype)

        ## reference net init
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=self.weight_dtype, device=device)
        self.reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
        )

        ## denoising net init
        if os.path.exists(config.motion_module_path):
            ### stage1 + stage2
            self.denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                config.pretrained_base_model_path,
                config.motion_module_path,
                subfolder="unet",
                unet_additional_kwargs=infer_config.unet_additional_kwargs,
            ).to(dtype=self.weight_dtype, device=device)
        else:
            ### only stage1
            self.denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                config.pretrained_base_model_path,
                "",
                subfolder="unet",
                unet_additional_kwargs={
                    "use_motion_module": False,
                    "unet_use_temporal_attention": False,
                    "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
                }
            ).to(dtype=self.weight_dtype, device=device)
        self.denoising_unet.load_state_dict(
            torch.load(config.denoising_unet_path, map_location="cpu"),
            strict=False
        )

        ## face locator init
        self.face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
            dtype=self.weight_dtype, device=device
        )
        self.face_locator.load_state_dict(torch.load(config.face_locator_path))

        ### load audio processor params
        self.audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

        ### load face detector params
        self.face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

        ############# model_init finished #############
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        self.scheduler = DDIMScheduler(**sched_kwargs)

        self.pipe = Audio2VideoPipeline(
            vae=self.vae,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            audio_guider=self.audio_processor,
            face_locator=self.face_locator,
            scheduler=self.scheduler,
        )
        self.pipe = self.pipe.to(device, dtype=self.weight_dtype)

    def run(self, input_image, input_audio_path, out_fpath):
        if self.config.seed is not None and self.config.seed > -1:
            generator = torch.manual_seed(self.config.seed)
        else:
            generator = torch.manual_seed(random.randint(100, 1000000))

        final_fps = self.config.fps

        #### face musk prepare
        face_img = np.array(input_image) #cv2.imread(ref_image_path)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')

        det_bboxes, probs = self.face_detector.detect(face_img)
        select_bbox = select_face(det_bboxes, probs)
        if select_bbox is None:
            face_mask[:, :] = 255
        else:
            xyxy = select_bbox[:4]
            xyxy = np.round(xyxy).astype('int')
            rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
            r_pad = int((re - rb) * self.config.facemusk_dilation_ratio)
            c_pad = int((ce - cb) * self.config.facemusk_dilation_ratio)
            face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255

            #### face crop
            r_pad_crop = int((re - rb) * self.config.facecrop_dilation_ratio)
            c_pad_crop = int((ce - cb) * self.config.facecrop_dilation_ratio)
            crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
            print(crop_rect)
            face_img, _ = crop_and_pad(face_img, crop_rect)
            face_mask, _ = crop_and_pad(face_mask, crop_rect)
            face_img = cv2.resize(face_img, (self.config.W, self.config.H))
            face_mask = cv2.resize(face_mask, (self.config.W, self.config.H))

        ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
        face_mask_tensor = torch.Tensor(face_mask).to(dtype=self.weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0

        video = self.pipe(
            ref_image_pil,
            input_audio_path,
            face_mask_tensor,
            self.config.W,
            self.config.H,
            self.config.L,
            self.config.steps,
            self.config.cfg,
            generator=generator,
            audio_sample_rate=self.config.sample_rate,
            context_frames=self.config.context_frames,
            fps=final_fps,
            context_overlap=self.config.context_overlap
        ).videos

        video = video
        save_videos_grid_moviepy(
            video,
            out_fpath,
            input_audio_path,
            n_rows=1,
            fps=final_fps,
        )