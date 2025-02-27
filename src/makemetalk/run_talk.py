import time
from typing import Optional
from omegaconf import OmegaConf
import argparse
import os

from fluxgenerator import FluxGenerator, modify_postpic_prompt
from speechanimator import setup_ffmpeg, SpeechAnimator
from promptmodifier import PromptModifier

from openai_appearance import get_appearance

from huggingface_hub import login
access_token_read = os.getenv("HUGGINGFACE_ACCESS_TOKEN_READ")
login(token = access_token_read)

setup_ffmpeg()

parser = argparse.ArgumentParser()
parser.add_argument("--config_flux", type=str, default="./configs/pulid_config.yaml")
parser.add_argument("--config_echo", type=str, default="./configs/prompts/animation_acc.yaml")
parser.add_argument("--id_images_path", type=str, default="/workspace/TestData/TestWomanAva")
parser.add_argument("--in_audio_path", type=str, default="/workspace/Data/Sel/Vesna-NePechalsaMashaPomogu_join.wav")
parser.add_argument("--user_prompt", type=str, default="in the city landscape, detailed face")
parser.add_argument("--out_fpath", type=str, default="/workspace/TestData/output/talking_me_2.mp4")
args = parser.parse_args()

config_flux = OmegaConf.load(args.config_flux)
config_echo = OmegaConf.load(args.config_echo)

#prepare out folder
os.makedirs(os.path.dirname(args.out_fpath), exist_ok=True)

generator = FluxGenerator(config_flux.name, config_flux.device, config_flux.offload, config_flux.aggressive_offload, config_flux)
speech_generator = SpeechAnimator(config_echo)
prompt_modifier = PromptModifier(config_flux.device)

#generate avatar embedding
id_embeddings, uncond_id_embeddings, gender = generator.generate_avatar_embedding(src_folder=args.id_images_path)
appearance = ""
if config_flux.use_appearance:
    appearance = get_appearance(id_image=args.id_images_path)

#p = modify_postpic_prompt(prompt=PROMPT, gender=gender, appearance=appearance)

p = prompt_modifier.make_prompt(gender=gender, user_prompt=args.user_prompt, appearance=appearance)
out_image, used_seed, intermediate = generator.generate_by_embeddings(
                        prompt=p,
                        id_embeddings=id_embeddings, 
                        uncond_id_embeddings=uncond_id_embeddings
            )
speech_generator.run(out_image, args.in_audio_path, args.out_fpath)



