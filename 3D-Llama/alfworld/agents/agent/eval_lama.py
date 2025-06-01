from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLaMACommandGenerator:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", device="cuda", save_name="lama-8b"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

        # Fix missing pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

        self.save_name = save_name
    
    def generata_prompt(self, obs, task):
        """
        Step 1: Select a role prompt based on task description.
        Step 2: [SEP] refers to previous action or observation, group them to previous actions.
        Step 3: Regulate user output to only one comaand with templates - alfred.twl2.
        """
        prompt = f""
        return prompt

    def command_generation_lama(self, observation_strings, task_desc_strings):
        res = []

        for obs, task in zip(observation_strings, task_desc_strings):
            # Construct prompt
            prompt = self.generata_prompt(obs, task)

            # Tokenize prompt (no padding needed for single input)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True).to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            # Generate response (small token limit for speed)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=1.0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and remove the prompt prefix
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # response_text = generated_text[len(prompt):].strip()
            
            # response_text = response_text.split("\n")[0].strip()

            res.append(generated_text)
            # res.append(response_text)

        return res, None  # current_dynamics is unused

if __name__ == "__main__":
    # Example usage
    generator = LLaMACommandGenerator(model_name="meta-llama/Meta-Llama-3-8B", device="cuda", save_name="lama-8b")
    
    observation_strings = ['-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 10, a cabinet 11, a cabinet 12, a cabinet 2, a cabinet 3, a cabinet 4, a cabinet 5, a cabinet 6, a cabinet 7, a cabinet 8, a cabinet 9, a coffeemachine 1, a countertop 1, a countertop 2, a diningtable 1, a drawer 1, a drawer 2, a drawer 3, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, and a toaster 1. [SEP] You arrive at cabinet 12. The cabinet 12 is closed. [SEP] go to cabinet 12 [SEP] You open the cabinet 12. The cabinet 12 is open. In it, you see a bowl 3. [SEP] open cabinet 12 [SEP] You close the cabinet 12. [SEP] close cabinet 12', '-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 10, a cabinet 11, a cabinet 12, a cabinet 13, a cabinet 14, a cabinet 15, a cabinet 16, a cabinet 17, a cabinet 18, a cabinet 19, a cabinet 2, a cabinet 20, a cabinet 21, a cabinet 22, a cabinet 23, a cabinet 24, a cabinet 25, a cabinet 26, a cabinet 3, a cabinet 4, a cabinet 5, a cabinet 6, a cabinet 7, a cabinet 8, a cabinet 9, a coffeemachine 1, a countertop 1, a countertop 2, a countertop 3, a drawer 1, a drawer 10, a drawer 11, a drawer 12, a drawer 2, a drawer 3, a drawer 4, a drawer 5, a drawer 6, a drawer 7, a drawer 8, a drawer 9, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, and a toaster 1. [SEP] You arrive at drawer 11. The drawer 11 is closed. [SEP] go to drawer 11 [SEP] You open the drawer 11. The drawer 11 is open. In it, you see nothing. [SEP] open drawer 11 [SEP] You close the drawer 11. [SEP] close drawer 11', '-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 2, a cabinet 3, a cabinet 4, a cabinet 5, a cabinet 6, a cabinet 7, a cabinet 8, a cabinet 9, a coffeemachine 1, a countertop 1, a countertop 2, a drawer 1, a drawer 10, a drawer 11, a drawer 12, a drawer 13, a drawer 2, a drawer 3, a drawer 4, a drawer 5, a drawer 6, a drawer 7, a drawer 8, a drawer 9, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, a stoveburner 5, a stoveburner 6, and a toaster 1. [SEP] You close the drawer 12. [SEP] close drawer 12 [SEP] You arrive at drawer 10. On the drawer 10, you see nothing. [SEP] go to drawer 10 [SEP] You arrive at drawer 12. The drawer 12 is closed. [SEP] go to drawer 12']
    task_desc_strings = ['put a cool plate in cabinet.', 'put two spatula in drawer.', 'put a clean butterknife in drawer.']
    Actions = ['go to cabinet 12', 'go to countertop 3', 'open drawer 12']
    
    
    commands, _ = generator.command_generation_lama(observation_strings, task_desc_strings)
    
    for cmd, baseline in zip(commands, Actions):
        print(f"Generated Command: {cmd} ")
        print(f"Baseline Command: {baseline}")

    # Example output:
