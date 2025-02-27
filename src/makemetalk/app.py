import gradio as gr
from typing import Optional
from omegaconf import OmegaConf
import argparse
import os
import numpy as np

from fluxgenerator import FluxGenerator, modify_postpic_prompt
from speechanimator import setup_ffmpeg, SpeechAnimator
from promptmodifier import PromptModifier

DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)

def create_demo(args):

    config_flux = OmegaConf.load(args.config_flux)
    config_echo = OmegaConf.load(args.config_echo)

    generator = FluxGenerator(config_flux.name, config_flux.device, config_flux.offload, config_flux.aggressive_offload, config_flux)
    speech_generator = SpeechAnimator(config_echo)
    prompt_modifier = PromptModifier(config_flux.device)

    with gr.Blocks(title="Make Me Talk", css=".gr-box {border-color: #8136e2}") as demo:
        #gr.Markdown(_HEADER_)
        with gr.Row():
            with gr.Column():
                num_references = gr.Textbox(label="Number of refereence persona images", value = "1")

                @gr.render(inputs=num_references)
                def show_image_boxes(text):
                    n_r = int(text)
                    if n_r <= 0:
                        gr.Markdown("## No Input Provided")
                    else:
                        for _ in range(n_r):
                            face_image = gr.Image(label="ID image (main)", sources="upload", type="numpy", height=256)
                with gr.Row():
                    face_image = gr.Image(label="ID image (main)", sources="upload", type="numpy", height=256)
                    supp_image1 = gr.Image(
                        label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                    )
                    supp_image2 = gr.Image(
                        label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                    )
                    supp_image3 = gr.Image(
                        label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                    )
                
                submit_avatar = gr.Button("Generate Avatar Embedding")

                prompt = gr.Textbox(label="Prompt", value='in the city landscape, detailed face')
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
                        H = gr.Slider(label="Height", value=1024, minimum=512, maximum=2024, step=64)
                        W = gr.Slider(label="Width", value=1024, minimum=512, maximum=2024, step=64)
                    with gr.Row():
                        id_scale = gr.Slider(label="ID scale", minimum=0, maximum=5, step=0.05, value=1.0, interactive=True)

                gr.Markdown("## Examples")
                example_inps = [
                    [
                        'portrait,cinematic,wolf ears,white hair',
                        'example_inputs/liuyifei.png',
                    ]
                ]
                gr.Examples(examples=example_inps, inputs=[prompt, face_image], label='realistic')

            with gr.Column():
                output = gr.Gallery(label='Output', elem_id="gallery")
                intermediate_output = gr.Gallery(label='DebugImage', elem_id="gallery", visible=False)

        inps_avatar = [
            face_image,
            supp_image1,
            supp_image2,
            supp_image3,
        ]
        inps_avatar = [ x for x in inps_avatar if x is not None]
        inps_avatar = [inps_avatar]

        inps = [
            prompt,
            audio,
            neg_prompt,
            scale,
            seed,
            steps,
            H,
            W,
            id_scale,
        ]

        submit_avatar.click(
            fn=generator.generate_avatar_embedding, 
            inputs=inps_avatar, 
            outputs=[]
            )

        #submit_video.click(
        #    fn=run, 
        #    inputs=inps, 
        #    outputs=[output, intermediate_output]
        #    )

    return demo


def create_demo_test(args):
    with gr.Blocks(title="Make Me Talk") as demo:
        #gr.Markdown(_HEADER_)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    face_image = gr.Image(label="ID image (main)", sources="upload", type="numpy", height=256)
                    supp_image1 = gr.Image(
                        label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                    )
                    supp_image2 = gr.Image(
                        label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                    )
                    supp_image3 = gr.Image(
                        label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                    )
    return demo


def create_demo_flux(args):
    import torch
    from einops import rearrange
    from PIL import Image
    config_flux = OmegaConf.load(args.config_flux)
    generator = FluxGenerator(config_flux.name, config_flux.device, config_flux.offload, config_flux.aggressive_offload, config_flux)

    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="portrait, color, cinematic")
                id_image = gr.Image(label="ID Image")
                id_weight = gr.Slider(0.0, 3.0, 1, step=0.05, label="id weight")

                width = gr.Slider(256, 1536, 896, step=16, label="Width")
                height = gr.Slider(256, 1536, 1152, step=16, label="Height")
                num_steps = gr.Slider(1, 20, 20, step=1, label="Number of steps")
                start_step = gr.Slider(0, 10, 0, step=1, label="timestep to start inserting ID")
                guidance = gr.Slider(1.0, 10.0, 4, step=0.1, label="Guidance")
                seed = gr.Textbox(-1, label="Seed (-1 for random)")
                max_sequence_length = gr.Slider(128, 512, 128, step=128,
                                                label="max_sequence_length for prompt (T5), small will be faster")

                with gr.Accordion("Advanced Options (True CFG, true_cfg_scale=1 means use fake CFG, >1 means use true CFG, if using true CFG, we recommend set the guidance scale to 1)", open=False):    # noqa E501
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="bad quality, worst quality, text, signature, watermark, extra limbs")
                    true_cfg = gr.Slider(1.0, 10.0, 1, step=0.1, label="true CFG scale")
                    timestep_to_start_cfg = gr.Slider(0, 20, 1, step=1, label="timestep to start cfg", visible=False)

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                seed_output = gr.Textbox(label="Used Seed")
                intermediate_output = gr.Gallery(label='Output', elem_id="gallery", visible=False)

        with gr.Row(), gr.Column():
                gr.Markdown("## Examples")
                example_inps = [
                    [
                        'a woman holding sign with glowing green text \"PuLID for FLUX\"',
                        'example_inputs/liuyifei.png',
                        4, 4, 2680261499100305976, 1
                    ],
                    [
                        'portrait, side view',
                        'example_inputs/liuyifei.png',
                        4, 4, 1205240166692517553, 1
                    ],
                    [
                        'white-haired woman with vr technology atmosphere, revolutionary exceptional magnum with remarkable details',  # noqa E501
                        'example_inputs/liuyifei.png',
                        4, 4, 6349424134217931066, 1
                    ],
                    [
                        'a young child is eating Icecream',
                        'example_inputs/liuyifei.png',
                        4, 4, 10606046113565776207, 1
                    ],
                    [
                        'a man is holding a sign with text \"PuLID for FLUX\", winter, snowing, top of the mountain',
                        'example_inputs/pengwei.jpg',
                        4, 4, 2410129802683836089, 1
                    ],
                    [
                        'portrait, candle light',
                        'example_inputs/pengwei.jpg',
                        4, 4, 17522759474323955700, 1
                    ],
                    [
                        'profile shot dark photo of a 25-year-old male with smoke escaping from his mouth, the backlit smoke gives the image an ephemeral quality, natural face, natural eyebrows, natural skin texture, award winning photo, highly detailed face, atmospheric lighting, film grain, monochrome',  # noqa E501
                        'example_inputs/pengwei.jpg',
                        4, 4, 17733156847328193625, 1
                    ],
                    [
                        'American Comics, 1boy',
                        'example_inputs/pengwei.jpg',
                        1, 4, 13223174453874179686, 1
                    ],
                    [
                        'portrait, pixar',
                        'example_inputs/pengwei.jpg',
                        1, 4, 9445036702517583939, 1
                    ],
                ]
                gr.Examples(examples=example_inps, inputs=[prompt, id_image, start_step, guidance, seed, true_cfg],
                            label='fake CFG')

                example_inps = [
                    [
                        'portrait, made of ice sculpture',
                        'example_inputs/lecun.jpg',
                        1, 1, 3811899118709451814, 5
                    ],
                ]
                gr.Examples(examples=example_inps, inputs=[prompt, id_image, start_step, guidance, seed, true_cfg],
                            label='true CFG')

        generate_btn.click(
            fn=generator.generate_image,
            inputs=[width, height, num_steps, start_step, guidance, seed, prompt, id_image, id_weight, neg_prompt,
                    true_cfg, timestep_to_start_cfg, max_sequence_length],
            outputs=[output_image, seed_output, intermediate_output],
        )

    return demo


class InterfaceModel():
    def __init__(self, args):
        self.persona_embedding = None
        config_flux = OmegaConf.load(args.config_flux)
        config_echo = OmegaConf.load(args.config_echo)

        self.generator = None #FluxGenerator(config_flux.name, config_flux.device, config_flux.offload, config_flux.aggressive_offload, config_flux)
        self.speech_generator = None #SpeechAnimator(config_echo)
        self.prompt_modifier = None #PromptModifier(config_flux.device)

    def dummy_avatar_generator(self, *args):
        src_list = []
        for el in args:
            if el is not None:
                src_list.append(el)
        return f"Avatar generated for {len(src_list)} reference images"

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

                        submit_avatar.click(
                            self.dummy_avatar_generator, 
                            face_images, 
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
                                    H = gr.Slider(label="Height", value=1024, minimum=512, maximum=2024, step=64)
                                    W = gr.Slider(label="Width", value=1024, minimum=512, maximum=2024, step=64)
                                with gr.Row():
                                    id_scale = gr.Slider(label="ID scale", minimum=0, maximum=5, step=0.05, value=1.0, interactive=True)

                    with gr.Column():
                        out_video = gr.PlayableVideo(label="Output")
                        spoken_text = gr.Textbox(label="Out text transcription", value='in the city landscape, detailed face', interactive=False)

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


    