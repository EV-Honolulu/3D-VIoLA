import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import tqdm

# Target model
# model_name = "/tmp2/rwliang/task_vector_8b_final"   # Local path
# model_name = "tatsu-lab/alpaca-8b"  # Hugging Face model
model_name = "meta-llama/Llama-3.2-8B"  # Hugging Face model

# Save file name
save_name = "meta-llama/Llama-3.2-3B"

# Load dataset and model
eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

# Store results here
results = []

# Loop over evaluation data
for example in tqdm.tqdm(eval_set, desc="Generating responses"):
    prompt = "BEGINNING OF CONVERSATION: USER: {" + example["instruction"] + "} ASSISTANT:"
    input_data = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=500
    )
    input_ids = input_data["input_ids"].to(device)
    attention_mask = input_data["attention_mask"].to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = generated_text[len(prompt):].strip()

    # Format result as requested
    result = {
        "instruction": example["instruction"],
        "output": response_text,
        "generator": save_name,
        "dataset": "selfinstruct",
        "datasplit": "eval"
    }
    results.append(result)

    # print(f"â
 Instruction: {example['instruction']}")
    # print(f"ð§  Output: {response_text}\n")

# Save to JSON file
with open(f"{save_name}.json", "w") as f:
    json.dump(results, f, indent=2)