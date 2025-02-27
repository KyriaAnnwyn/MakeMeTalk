import gradio as gr
from typing import Optional
from omegaconf import OmegaConf
import argparse
import os
import numpy as np

from fluxgenerator import FluxGenerator, modify_postpic_prompt
from speechanimator import setup_ffmpeg, SpeechAnimator
from promptmodifier import PromptModifier

from openai_appearance import get_appearance

DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)

class InterfaceModel():
    def __init__(self, args):
        self.persona_embedding = None
        config_flux = OmegaConf.load(args.config_flux)
        config_echo = OmegaConf.load(args.config_echo)

        self.generator = None #FluxGenerator(config_flux.name, config_flux.device, config_flux.offload, config_flux.aggressive_offload, config_flux)
        self.speech_generator = None #SpeechAnimator(config_echo)
        self.prompt_modifier = None #PromptModifier(config_flux.device)

        self.appearance = ""
        self.id_embeddings = None
        self.uncond_id_embeddings = None 
        self.gender = None

    def dummy_avatar_generator(self, *args):
        src_list = []
        for el in args[:-1]:
            if el is not None:
                src_list.append(el)
        use_appearance =args[-1]

        return f"Avatar generated for {len(src_list)} reference images. Use_appearance = {use_appearance}"
    
    def dummy_video_generator(self, *args):
        user_prompt = args[0]

        return f"Video generated with user prompt = {user_prompt}"
    
    def avatar_generator(self, *args):
        src_list = []
        for el in args[:-1]:
            if el is not None:
                src_list.append(el)
        use_appearance =args[-1]

        self.id_embeddings, self.uncond_id_embeddings, self.gender = self.generator.generate_avatar_embedding(src_folder=src_list)

        if use_appearance:
            self.appearance = get_appearance(id_image=src_list)

        return f"Avatar generated for {len(src_list)} reference images. Use_appearance = {use_appearance}"
    
    def video_generator(self, *args):
        user_prompt = args[0]

        p = self.prompt_modifier.make_prompt(gender=self.gender, user_prompt=user_prompt, appearance=self.appearance)
        out_image, used_seed, intermediate = self.generator.generate_by_embeddings(
                        prompt=p,
                        id_embeddings=self.id_embeddings, 
                        uncond_id_embeddings=self.uncond_id_embeddings
            )
        self.speech_generator.run(out_image, args.in_audio_path, args.out_fpath)
        return f"Video generated"

    def create_demo(self, args):

        with gr.Blocks(title="Make Me Talk", css=".gr-box {border-color: #8136e2}") as demo:
            #gr.Markdown(_HEADER_)
            with gr.Group():
                with gr.Column():
                    num_references = gr.Number(value=1, label="Number of reference persona images")

                    @gr.render(inputs=num_references)
                    def show_image_boxes(images_count):
                        face_images = []

                        if images_count <= 0:
                            gr.Markdown("## You selected 0 for reference images, should >= 1")
                        else:
                            with gr.Row():
                                for i in range(images_count):
                                    face_image = gr.Image(key=i, label=f"ID image {i}", sources="upload", type="numpy", height=256)
                                    face_images.append(face_image)

                        inputs = face_images + [use_appearance]
                        submit_avatar.click(
                            self.dummy_avatar_generator, 
                            inputs, 
                            output
                            )
        
                    use_appearance = gr.Checkbox(label="Use OpenAI to generate more accurate appearance"),
                    submit_avatar = gr.Button("Generate Avatar Embedding")
                    output = gr.Textbox(label="Avatar generation status", interactive=False)

            with gr.Group():
                with gr.Row():
                    
                    with gr.Column():
                            prompt = gr.Textbox(label="Prompt", value='in the city landscape, detailed face', interactive=True)
                            audio = gr.Audio(label="audio file with speech", sources="upload")

                            submit_video = gr.Button("Generate story")

                            with gr.Accordion("Advanced Options)", open=False):    # noqa E501
                                neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
                                scale = gr.Slider(
                                    label="CFG, recommend value range [1, 1.5], 1 will be faster ",
                                    value=1.2,
                                    minimum=1,
                                    maximum=1.5,
                                    step=0.1,
                                )
                                seed = gr.Slider(
                                    label="Seed", value=42, minimum=np.iinfo(np.uint32).min, maximum=np.iinfo(np.uint32).max, step=1
                                )
                                steps = gr.Slider(label="Steps", value=4, minimum=1, maximum=100, step=1)
                                with gr.Row():
                                    H = gr.Slider(label="Height", value=1024, minimum=512, maximum=1280, step=64)
                                    W = gr.Slider(label="Width", value=1024, minimum=512, maximum=1280, step=64)
                                with gr.Row():
                                    id_scale = gr.Slider(label="ID scale", minimum=0, maximum=5, step=0.05, value=1.0, interactive=True)

                    with gr.Column():
                        out_video = gr.PlayableVideo(label="Output")
                        #spoken_text = gr.Textbox(label="Out text transcription", value='in the city landscape, detailed face', interactive=False)

        return demo

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_flux", type=str, default="./configs/pulid_config.yaml")
    parser.add_argument("--config_echo", type=str, default="./configs/prompts/animation_acc.yaml")
    parser.add_argument("--port", type=int, default=8081, help="Port to use")
    args = parser.parse_args()

    processor = InterfaceModel(args)
    demo = processor.create_demo(args)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


    