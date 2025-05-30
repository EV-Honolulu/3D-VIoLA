from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLaMACommandGenerator:
    def __init__(self, model_name="meta-llama/Llama-2-8b-chat-hf", device="cuda", save_name="lama-8b"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
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
            
            # Tokenize prompt
            input_data = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=500
            )
            input_ids = input_data["input_ids"].to(self.device)
            attention_mask = input_data["attention_mask"].to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,
                    temperature=0.7,
                    top_p=1.0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text[len(prompt):].strip()
            res.append(response_text)

        return res, None  # current_dynamics is not used

