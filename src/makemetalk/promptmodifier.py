from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class PromptModifier():
    def __init__(self, device: str, model_path = "HuggingFaceH4/zephyr-7b-alpha"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Most LLMs don't have a pad token by default
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", load_in_4bit=True
        )
        self.device = device

        self.system_prompt = "You need to rewrite image prompt for diffusion generation. The output should be short, but contain all needed information: prompt should aim at realistic portrait photo generation"
        self.get_gender_system = "Read the text prompt and determine if the main figure is male or female"
        self.sample_prompt = "portrait photo of a person"

    def make_prompt(self, gender: str, user_prompt: str, appearance: str):
        input_prompt = gender + " " + appearance + " " + self.sample_prompt + " " + user_prompt

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": input_prompt},
        ]

        # By default, the output will contain up to 20 tokens
        model_inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        input_length = model_inputs.shape[1]
        generated_ids = self.model.generate(model_inputs, do_sample=False, max_new_tokens=50)
        return self.tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
    
    def get_gender_from_prompt(self, prompt : str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.get_gender_system,
            },
            {"role": "user", "content": prompt},
        ]

        # By default, the output will contain up to 20 tokens
        model_inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        input_length = model_inputs.shape[1]
        generated_ids = self.model.generate(model_inputs, do_sample=False, max_new_tokens=10)
        result = self.tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
        if "female" in result:
            return "female"

        return "male" 

    def generate_appearance4prompt(self, prompt: str) -> str:
        pass
    
if __name__ == "__main__":
    pm = PromptModifier(device = "cuda")

    gender = pm.get_gender_from_prompt(prompt = "beautiful red thin woman")

    print(gender)