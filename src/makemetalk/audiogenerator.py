import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import os

class AudioGenerator():
    def __init__(self, device: str, model_path = "Zyphra/Zonos-v0.1-hybrid"):
        self.model = Zonos.from_pretrained(model_path, device=device)
        self.device = device
        self.speaker = None

    def update_speaker(self, sample_voice_file) -> None: 
        wav, sampling_rate = torchaudio.load(sample_voice_file)
        self.speaker = self.model.make_speaker_embedding(wav, sampling_rate)   

    def generate_audio_speech(self, text: str) -> str:
        cond_dict = make_cond_dict(text=text, speaker=self.speaker, language="en-us")
        conditioning = self.model.prepare_conditioning(cond_dict)

        codes = self.model.generate(conditioning)

        wavs = self.model.autoencoder.decode(codes).cpu()

        os.makedirs("tmp", exist_ok=True)
        torchaudio.save("tmp/sample.wav", wavs[0], self.model.autoencoder.sampling_rate)

if __name__ == "__main__":
    ag = AudioGenerator(device = "cuda")
    ag.update_speaker(sample_voice_file="assets/sample_audio.wav")

    ag.generate_audio_speech(text="I love cats. I have a beautiful cat. She likes to play.")

   