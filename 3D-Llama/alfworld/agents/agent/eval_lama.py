from pathlib import Path
import transformers
import torch
import os
from llama_cpp import Llama


# if not Path('./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf').exists():
#     !wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf

# os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
# set device to cuda if available, else cpu
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.cuda.set_device(2)

# Load the model onto GPU
# llama3 = Llama(
#     "./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
#     verbose=False,
#     n_gpu_layers=-1,
#     n_ctx=32768,    # This argument is how many tokens the model can take. The longer the better, but it will consume more memory. 16384 is a proper value for a GPU with 16GB VRAM.
# )

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def generate_response(_model, _messages: str) -> str:
    '''
    This function will inference the model with given messages.
    '''
    _output = _model.create_chat_completion(
        _messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=512,    # This argument is how many tokens the model can generate, you can change it and observe the differences.
        temperature=0,      # This argument is the randomness of the model. 0 means no randomness. You will get the same result with the same input every time. You can try to set it to different values.
        repeat_penalty=2.0,
    )["choices"][0]["message"]["content"]
    return _output

class LLMAgent():
    def __init__(self, role_description: str, task_description: str, pipeline, llm:str="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"):
        self.role_description = role_description   # Role means who this agent should act like. e.g. the history expert, the manager......
        self.task_description = task_description    # Task description instructs what task should this agent solve.
        self.llm = llm  # LLM indicates which LLM backend this agent is using.
        self.pipeline = pipeline

    @classmethod
    def read_my_file(cls):
        # Read "role.txt" from the same folder as this script
        try:
            folder = Path(__file__).parent  # works only in .py files
        except NameError:
            folder = Path.cwd()             # fallback for notebooks or interactive mode
        file_path = folder / "role.txt"
        text = file_path.read_text()
        # 2. assign to cls.role_prompt
        cls.role_text = text

        # 3. assign to cls.action_text
        action_file_path = folder / "action.txt"
        if action_file_path.exists():
            cls.action_text = action_file_path.read_text().strip()
        else:
            cls.action_text = "You are a helpful assistant. Please provide a single command based on the task and observations."

    def generate_prompt(self, obs, task):
        """
        Step 1: Select a role prompt based on task description. -> done
        Step 2: [SEP] refers to previous action or observation, group them to previous actions.
        Step 3: Regulate user output to only one command with templates - alfred.twl2.
        """
        
        # Step 1: select role prompt based on task description
        if "put a" in task or "put some" in task:
            # Pick & Place
            role_prompt = self.__class__.role_text.split("**Pick & Place**")[1].split("---")[0].strip()
        elif "look at" in task or "examine" in task:
            # Examine in Light
            role_prompt = self.__class__.role_text.split("**Examine in Light**")[1].split("---")[0].strip()
        elif "clean" in task and "put" in task:
            # Clean & Place
            role_prompt = self.__class__.role_text.split("**Clean & Place**")[1].split("---")[0].strip()
        elif "heat" in task and "put" in task:
            # Heat & Place
            role_prompt = self.__class__.role_text.split("**Heat & Place**")[1].split("---")[0].strip()
        elif "cool" in task and "put" in task:
            # Cool & Place
            role_prompt = self.__class__.role_text.split("**Cool & Place**")[1].split("---")[0].strip()
        elif "put two" in task or "find two" in task:
            # Pick Two & Place
            role_prompt = self.__class__.role_text.split("**Pick Two & Place**")[1].split("---")[0].strip()
        else:
            role_prompt = ""  # fallback or raise an error/log warning

        role = f"{role_prompt}\n\n"
        role += self.__class__.action_text + "\n"
        role += "Choose one action for next step.\n"
        self.role_description = role 

        task_prompt = f"\nTask: {task}\n"
        self.task_description = task_prompt  # Store the task description for later use

        # Step 2: Group previous actions and observations
        obs_split = obs.split("[SEP]")
        env_prompt = obs_split[0].strip()  # Initial environment description
        observations = [s.strip() for s in obs_split[1:] if s.strip()]  # Filter out empty strings
        
        
        prompt = f"Environment: {env_prompt}\n"
        prompt += f"\nTask: {task}\n"
        prompt += "Previous Actions and Observations:\n"
        for i, observation in enumerate(observations):
            if i % 2 == 0:
                prompt += f"Observation {i//2 + 1}: {observation}\n"
            else:
                prompt += f"Action {i//2 + 1}: {observation}\n"
        prompt += f"Please answer with one of the following actions only.\
                    Format your answer as: **action: [action]** \
                    Examples: action: go to fridge 1, action: open drawer 2, action: take plate from drawer 1"
        return prompt


    
    def inference(self, observation_strings, task_desc_strings) -> list:
        res = []
        read_my_file = self.__class__.read_my_file
        if not hasattr(self.__class__, 'role_text') or not self.__class__.role_text:
            read_my_file()
        if not self.__class__.role_text:
            raise ValueError("Role text not loaded. Please ensure 'role.txt' is present in the same directory as this script.")
        
        for obs, task in zip(observation_strings, task_desc_strings):
            message = self.generate_prompt(obs, task)
            messages = [
                {"role": "system", "content": f"{self.role_description}"},  # Hint: you may want the agents to speak Traditional Chinese only.
                {"role": "user", "content": f"{self.task_description}\n{message}"}, # Hint: you may want the agents to clearly distinguish the task descriptions and the user messages. A proper seperation text rather than a simple line break is recommended.
            ]
            # generated_text =  generate_response(llama3, messages)
            generated_text =  self.pipeline(
                messages,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            generated_text = generated_text[0]['generated_text']
            assistant_msg = next(msg for msg in generated_text if msg["role"] == "assistant")["content"].replace("**", "").strip()
            # print(assistant_msg)
            res.append(assistant_msg)
        return res, None


if __name__ == "__main__":
    observation_strings = ['-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 10, a cabinet 11, a cabinet 12, a cabinet 2, a cabinet 3, a cabinet 4, a cabinet 5, a cabinet 6, a cabinet 7, a cabinet 8, a cabinet 9, a coffeemachine 1, a countertop 1, a countertop 2, a diningtable 1, a drawer 1, a drawer 2, a drawer 3, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, and a toaster 1. [SEP] You arrive at cabinet 12. The cabinet 12 is closed. [SEP] go to cabinet 12 [SEP] You open the cabinet 12. The cabinet 12 is open. In it, you see a bowl 3. [SEP] open cabinet 12 [SEP] You close the cabinet 12. [SEP] close cabinet 12', '-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 10, a cabinet 11, a cabinet 12, a cabinet 13, a cabinet 14, a cabinet 15, a cabinet 16, a cabinet 17, a cabinet 18, a cabinet 19, a cabinet 2, a cabinet 20, a cabinet 21, a cabinet 22, a cabinet 23, a cabinet 24, a cabinet 25, a cabinet 26, a cabinet 3, a cabinet 4, a cabinet 5, a cabinet 6, a cabinet 7, a cabinet 8, a cabinet 9, a coffeemachine 1, a countertop 1, a countertop 2, a countertop 3, a drawer 1, a drawer 10, a drawer 11, a drawer 12, a drawer 2, a drawer 3, a drawer 4, a drawer 5, a drawer 6, a drawer 7, a drawer 8, a drawer 9, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, and a toaster 1. [SEP] You arrive at drawer 11. The drawer 11 is closed. [SEP] go to drawer 11 [SEP] You open the drawer 11. The drawer 11 is open. In it, you see nothing. [SEP] open drawer 11 [SEP] You close the drawer 11. [SEP] close drawer 11', '-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a cabinet 2, a cabinet 3, a cabinet 4, a cabinet 5, a cabinet 6, a cabinet 7, a cabinet 8, a cabinet 9, a coffeemachine 1, a countertop 1, a countertop 2, a drawer 1, a drawer 10, a drawer 11, a drawer 12, a drawer 13, a drawer 2, a drawer 3, a drawer 4, a drawer 5, a drawer 6, a drawer 7, a drawer 8, a drawer 9, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 1, a stoveburner 2, a stoveburner 3, a stoveburner 4, a stoveburner 5, a stoveburner 6, and a toaster 1. [SEP] You close the drawer 12. [SEP] close drawer 12 [SEP] You arrive at drawer 10. On the drawer 10, you see nothing. [SEP] go to drawer 10 [SEP] You arrive at drawer 12. The drawer 12 is closed. [SEP] go to drawer 12']
    task_desc_strings = ['put a cool plate in cabinet.', 'put two spatula in drawer.', 'put a clean butterknife in drawer.']
    Actions = ['go to cabinet 12', 'go to countertop 3', 'open drawer 12']


    emboddied_agent = LLMAgent(role_description="", task_description="", pipeline=pipeline)
    commands,_ = emboddied_agent.inference(observation_strings, task_desc_strings)

    for cmd, baseline in zip(commands, Actions):
        print(f"Generated Command: {cmd} ")
        print(f"Baseline Command: {baseline}\n")