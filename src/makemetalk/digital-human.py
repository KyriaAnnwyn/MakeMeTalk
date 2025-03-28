import gradio as gr
from typing import Optional
from omegaconf import OmegaConf
import argparse
import os
import numpy as np
import cv2
import torchaudio
import torch

from fluxgenerator import FluxGenerator, modify_postpic_prompt
from speechanimator import setup_ffmpeg, SpeechAnimator
from promptmodifier import PromptModifier
from audiogenerator import AudioGenerator

from openai_appearance import get_appearance
from operate_user_bio import generate_BIO
from openai_lifestory import generate_story_prompts

REAL_MODE = False

DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)

_HEADER_AVATAR_ = '''
<h2><b><center>Generate your persona using images or description</center></b></h2>

'''

_HEADER_STORY_ = '''
<h2><b><center>Generate your talking story</center></b></h2>

'''

SAMPLE_BIO = "Olivia Silverleaves is a 26-year-old lifestyle and psychology blogger from Los Angeles, California, captivating an audience primarily of men seeking guidance in their romantic relationships. She started to write blog at 26 years old, 2024, 1 of May.  Embracing her flair for communication and her own tumultuous experiences, Olivia carved out a niche in psychology through self-education and a profound personal journey, rather than traditional academic routes. Her blog began as a personal project, spurred by a deeply transformative episode in her own love life. Olivia's story started with a romance that seemed destined for a fairy tale ending but instead concluded on a somber note. This heartbreak was not the end for Olivia; it was the catalyst for her exploration into the complexities of human emotions and relationships. Through her posts, she combines insightful psychological concepts with practical advice, all while maintaining a charismatic and relatable tone. Olivia's mission is to provide a platform that not only explores the intricacies of male-female dynamics but also offers a beacon of hope and strategies for those navigating the challenging waters of love and relationships. Her approachable style and honest reflections have made her a beloved figure among her followers, who appreciate not just the advice she offers but the genuine care with which she delivers it. Also Olivia is an ordinary attractive lady. She likes fitness, reading psychology books, walking with her friends, parties, shopping and every things an ordinary girls likes. She likes to make provocative photos to attract men to her blog."



class InterfaceModel():
    def __init__(self, args):
        self.persona_embedding = None
        config_flux = OmegaConf.load(args.config_flux)
        config_echo = OmegaConf.load(args.config_echo)

        self.generator = None 
        self.speech_generator = None 
        self.prompt_modifier = None 
        self.audio_generator = None

        if REAL_MODE:
            self.generator = FluxGenerator(config_flux.name, config_flux.device, config_flux.offload, config_flux.aggressive_offload, config_flux)
            self.speech_generator = SpeechAnimator(config_echo)
            self.prompt_modifier = PromptModifier(config_flux.device)
            self.audio_generator = AudioGenerator(config_flux.device)

        self.appearance = ""
        self.id_embeddings = None
        self.uncond_id_embeddings = None 
        self.gender = None
        self.voice_sample = None
        self.bio = None

    def dummy_avatar_generator(self, *args):
        src_list = []
        for el in args[:-1]:
            if el is not None:
                src_list.append(el)
        use_appearance =args[-1]

        status_str = f"Avatar generated for {len(src_list)} reference images. Use_appearance = {use_appearance}"
        return status_str
    
    def dummy_avatar_generator_by_text(self, *args):
        prompt = args[0]
        
        self.appearance = ""
        size = (w, h, channels) = (1024, 1024, 3)
        image = np.zeros(size, np.uint8)

        status_str = f"Avatar generated for generated reference image. Appearance = {self.appearance}"
        return image, status_str

    def dummy_create_persona(self, *args):
        bio = args[0]
        self.voice_sample = args[1]
        user_full_name = args[2]

        self.bio = generate_BIO(user_full_name = user_full_name, user_description = bio, gender = self.gender)
        return "Persona created"
    
    def dummy_audio_generator(self, *args):
        text = args[0]
        voice_sample = self.voice_sample

        gen_audio = "assets/sample_audio.wav"
        return gen_audio
    
    def dummy_video_generator(self, *args):
        user_prompt = args[0]
        video_path = "assets/talking_me_2.mp4"

        return f"Video generated with user prompt = {user_prompt}", video_path

    def avatar_generator(self, *args):
        print("Generating avatar from images")
        src_list = []
        for el in args[:-1]:
            if el is not None:
                src_list.append(el)
        use_appearance =args[-1]

        self.id_embeddings, self.uncond_id_embeddings, self.gender = self.generator.generate_avatar_embedding(src_folder=src_list)

        if use_appearance:
            self.appearance = get_appearance(id_image=src_list)

        return f"Avatar generated for {len(src_list)} reference images. Use_appearance = {use_appearance}"
    
    def avatar_generator_by_text(self, *args):
        prompt = args[0]
        
        #generate appearance for this prompt
        self.appearance = ""
        self.gender = self.prompt_modifier.get_gender_from_prompt(prompt)
        #enhance prompt + appearance with details
        p = self.prompt_modifier.make_prompt(gender=self.gender, user_prompt=prompt, appearance=self.appearance)
        #generate image for this prompt
        image, _, _ = self.generator.generate_image(
                        prompt=p
            )
        image = np.array(image)
        #generate face embeddings
        self.id_embeddings, self.uncond_id_embeddings, self.gender = self.generator.generate_avatar_embedding(src_folder=[image])

        status_str = f"Avatar generated for generated reference image. Appearance = {self.appearance}"
        return image, status_str
    
    def convert_audio(self, audio: tuple) -> tuple:
        sr, y = audio
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        y = torch.from_numpy(y)
        y = y.swapaxes(0,1)
        return (sr, y)

    def create_persona(self, *args):
        bio = args[0]
        self.voice_sample = args[1]
        user_full_name = args[2]

        self.bio = generate_BIO(user_full_name = user_full_name, user_description = bio, gender = self.gender)

        self.voice_sample = self.convert_audio(self.voice_sample)
        torchaudio.save("tmp/voice_sample.wav", self.voice_sample[1], self.voice_sample[0])
        
        self.audio_generator.update_speaker(self.voice_sample)
        return "Persona created"
    
    def audio_generator_function(self, *args):
        text = args[0]

        gen_audio = self.audio_generator.generate_audio_speech(text=text)
        #gen_audio = self.audio_generator.generate_audio_speech_tuple(text=text)
        return gen_audio
    
    def video_generator(self, *args):
        user_prompt = args[0]
        in_audio = args[1]

        in_audio_path = "tmp/in_audio.wav"
        os.makedirs("tmp", exist_ok=True)

        in_audio = self.convert_audio(in_audio)
        torchaudio.save(in_audio_path, in_audio[1], in_audio[0])
        out_fpath = "tmp/my_talking_story.mp4"
        os.makedirs("tmp", exist_ok=True)

        p = self.prompt_modifier.make_prompt(gender=self.gender, user_prompt=user_prompt, appearance=self.appearance)
        out_image, used_seed, intermediate = self.generator.generate_by_embeddings(
                        prompt=p,
                        id_embeddings=self.id_embeddings, 
                        uncond_id_embeddings=self.uncond_id_embeddings
            )
        self.speech_generator.run(out_image, in_audio_path, out_fpath)
        return f"Video generated", out_fpath

    def story_facts_generator(self, *args):

        story, bgr = generate_story_prompts(user_full_name = self.bio.get_user_name())

        return bgr, story

    def create_demo(self, args):

        with gr.Blocks(title="Make Me Talk", css=".gr-box {border-color: #8136e2}") as demo:
            #gr.Markdown(_HEADER_)
            with gr.Group():
                gr.Markdown(_HEADER_AVATAR_)
                with gr.Column():
                    with gr.Tab("Use reference images"):
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
                            if REAL_MODE:
                                submit_avatar.click(
                                    self.avatar_generator, 
                                    inputs, 
                                    output
                                    )
                            else:
                                submit_avatar.click(
                                    self.dummy_avatar_generator, 
                                    inputs, 
                                    output
                                    )
            
                        use_appearance = gr.Checkbox(label="Use OpenAI to generate more accurate appearance")
                        submit_avatar = gr.Button("Generate Avatar Embedding")
                        output = gr.Textbox(label="Avatar generation status", interactive=False)
                    with gr.Tab("Use description images"):
                        prompt = gr.Textbox(label="Persona description", value='beautiful red thin woman', interactive=True)
                        submit_textavatar = gr.Button("Generate Avatar Image and Embedding")
                        output_satus = gr.Textbox(label="Avatar generation status", interactive=False)
                        output_image = gr.Image(label=f"Genrated ID image", type="numpy", height=256, interactive=False)

                        inputs = [prompt]
                        if REAL_MODE:
                            submit_textavatar.click(
                                    self.avatar_generator_by_text, 
                                    inputs, 
                                    [output_image, output_satus]
                                    )
                        else:
                            submit_textavatar.click(
                                    self.dummy_avatar_generator_by_text, 
                                    inputs, 
                                    [output_image, output_satus]
                                    )
                        
                persona_name = gr.Textbox(label="Your name", value="Olivia Silverleaves", interactive=True)
                bio = gr.Textbox(label="Your interests, lifestyle, any necessary info about you", value=SAMPLE_BIO, interactive=True)
                voice_sample = gr.Audio(label="audio file with your voice", sources="upload")
                submit_persona = gr.Button("Generate your Persona")
                persona_creation_status = gr.Textbox(label="Persona creation status", value='Not created yet', interactive=False)

                inputs = [bio, voice_sample, persona_name]
                if REAL_MODE:
                    submit_persona.click(
                        self.create_persona,
                        inputs, 
                        [persona_creation_status]
                    )
                else:
                    submit_persona.click(
                        self.dummy_create_persona, 
                        inputs, 
                        [persona_creation_status]
                    )
                
            with gr.Group():
                gr.Markdown(_HEADER_STORY_)
                with gr.Row():                   
                    with gr.Column():
                        video_inputs = []
                        with gr.Tab("Auto generated story"):
                            submit_gen_story_prompts = gr.Button("Auto generate you story for today")

                            prompt = gr.Textbox(label="Prompt", value='in the city landscape, detailed face', interactive=True)
                            text2speak = gr.Textbox(label="Text you want to speak", value='My cat came to me just recently, it was very strange, I was just walking down the street and when I approached the store I saw a black cat, he ran up to me and started purring.', interactive=True)

                            inputs_autogen = []
                            submit_gen_story_prompts.click(
                                self.story_facts_generator,
                                inputs_autogen,
                                [prompt, text2speak]
                            )
    
                            submit_gen_audio = gr.Button("Generate Audio for the speech text")
                            audio = gr.Audio(label="speech audio")

                            inputs = [text2speak]
                            if REAL_MODE:
                                submit_gen_audio.click(
                                    self.audio_generator_function,
                                    inputs, 
                                    [audio]
                                )
                            else:
                                submit_gen_audio.click(
                                    self.dummy_audio_generator, 
                                    inputs, 
                                    [audio]
                                )

                            video_inputs = [prompt, audio]
                           
                        with gr.Tab("User defined story"):
                            prompt = gr.Textbox(label="Prompt", value='in the city landscape, detailed face', interactive=True)
                            mode = gr.Radio(["use audio", "use text"], value="textbox")

                            @gr.render(inputs=[mode], triggers=[mode.input])
                            def show_split(mode):
                                if mode == "use audio":
                                    audio = gr.Audio(label="audio file with speech", sources=["upload", "microphone"])
                                else:
                                    text2speak = gr.Textbox(label="Text you want to speak", value='My cat came to me just recently, it was very strange, I was just walking down the street and when I approached the store I saw a black cat, he ran up to me and started purring.', interactive=True)
                                    submit_gen_audio = gr.Button("Generate Audio for the speech text")
                                    audio = gr.Audio(label="speech audio")

                                    inputs = [text2speak]
                                    if REAL_MODE:
                                        submit_gen_audio.click(
                                            self.audio_generator_function,                                        
                                            inputs, 
                                            [audio]
                                            ) 
                                    else:  
                                        submit_gen_audio.click(                                     
                                            self.dummy_audio_generator, 
                                            inputs, 
                                            [audio]
                                            )        

                                video_inputs = [prompt, audio]
                            
                        submit_video = gr.Button("Generate story")
                        
                    with gr.Column():
                        out_video = gr.PlayableVideo(label="Output", interactive=False)
                        status_vg = gr.Textbox(label="Avatar generation status", interactive=False)

                        if REAL_MODE:
                            submit_video.click(
                                self.video_generator,
                                video_inputs, 
                                [status_vg, out_video]
                            )
                        else:
                            submit_video.click(
                                self.dummy_video_generator, 
                                video_inputs, 
                                [status_vg, out_video]
                            )

        return demo

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_flux", type=str, default="./configs/pulid_config.yaml")
    parser.add_argument("--config_echo", type=str, default="./configs/prompts/animation_acc.yaml")
    parser.add_argument("--port", type=int, default=8080, help="Port to use")
    args = parser.parse_args()

    processor = InterfaceModel(args)
    demo = processor.create_demo(args)
    demo.launch(server_name='0.0.0.0', server_port=args.port, share=True)


    