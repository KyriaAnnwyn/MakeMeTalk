import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import os

MODEL_PATH = "Zyphra/Zonos-v0.1-transformer" #"Zyphra/Zonos-v0.1-hybrid"

class AudioGenerator():
    def __init__(self, device: str, model_path = MODEL_PATH):
        self.model = Zonos.from_pretrained(model_path, device=device)
        self.device = device
        self.speaker = None

    def update_speaker(self, sample_voice_file: str | tuple) -> None: 
        if isinstance(sample_voice_file, str):
            wav, sampling_rate = torchaudio.load(sample_voice_file)
        else:
            wav, sampling_rate = sample_voice_file[1], sample_voice_file[0]
        self.speaker = self.model.make_speaker_embedding(wav, sampling_rate)   

    def generate_audio_speech(self, text: str) -> str:
        cond_dict = make_cond_dict(text=text, speaker=self.speaker, language="en-us")
        conditioning = self.model.prepare_conditioning(cond_dict)

        codes = self.model.generate(conditioning)
        wavs = self.model.autoencoder.decode(codes).cpu()
        os.makedirs("tmp", exist_ok=True)

        #make it dualchannel
        w  = torch.squeeze(wavs[0])
        w = torch.stack((w,w))
        torchaudio.save("tmp/sample.wav", w, self.model.autoencoder.sampling_rate)

        return "tmp/sample.wav"

    def generate_audio_speech_tuple(self, text: str) -> tuple:
        cond_dict = make_cond_dict(text=text, speaker=self.speaker, language="en-us")
        conditioning = self.model.prepare_conditioning(cond_dict)

        codes = self.model.generate(conditioning)
        wavs = self.model.autoencoder.decode(codes).cpu()

        os.makedirs("tmp", exist_ok=True)
        #make it dualchannel
        w  = torch.squeeze(wavs[0])
        w = torch.stack((w,w))
        return (self.model.autoencoder.sampling_rate, wavs[0])

if __name__ == "__main__":
    ag = AudioGenerator(device = "cuda")
    ag.update_speaker(sample_voice_file="assets/sample_audio.wav")

    ag.generate_audio_speech(text="I love cats. I have a beautiful cat. She likes to play.")

   