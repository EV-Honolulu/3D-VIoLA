from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLaMACommandGenerator:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", device="cuda", save_name="lama-8b"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

        # Fix missing pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

        self.save_name = save_name

    def command_generation_lama(self, observation_strings, task_desc_strings):
        res = []

        for obs, task in zip(observation_strings, task_desc_strings):
            # Construct prompt
            prompt = f"BEGINNING OF CONVERSATION: USER: {{Task: {task} | Observation: {obs}}} ASSISTANT:"

            # Tokenize prompt (no padding needed for single input)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            # Generate response (small token limit for speed)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=64,
                    temperature=0.7,
                    top_p=1.0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and remove the prompt prefix
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response_text = generated_text[len(prompt):].strip()

            # (Optional) keep just the first line if more than one line generated
            response_text = response_text.split("\n")[0].strip()

            res.append(response_text)

        return res, None  # current_dynamics is unused


