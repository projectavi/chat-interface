from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, pipeline
import torch


class ChatGenerator:

    def __init__(self, hf_key):
        self.temperature = 1.0
        self.repetition_penalty = 1.5

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.hf_key = hf_key

        self.model_config = AutoConfig.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_key
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            config=self.model_config,
            quantization_config=self.bnb_config,
            device_map="auto",
            use_auth_token=self.hf_key
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_key
        )

        self.pipeline = pipeline(
            model=self.model, tokenizer=self.tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=self.temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=self.repetition_penalty  # without this output begins repeating
        )

    def generate(self, message):
        return self.pipeline(message)[0]["generated_text"]


if __name__ == "__main__":
    hf_key = input("Input your HF Key: ")
    generator = ChatGenerator(hf_key)

    while (True):
        prompt = input()
        print(generator.generate(prompt))
