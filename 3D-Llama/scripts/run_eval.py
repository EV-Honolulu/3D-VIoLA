import os
import json
import importlib
import logging

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.agent import TextDAggerAgent
from alfworld.agents.eval import evaluate_dagger, evaluate_dqn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_eval():
    config = generic.load_config()
    agent = TextDAggerAgent(config)

    output_dir = os.path.expandvars(config["general"]["save_path"])
    save_output_dir = os.path.join(config["general"]["evaluate"]["eval_folder_save"])
    num_files_in_save_output_dir = len(os.listdir(save_output_dir)) if os.path.exists(output_dir) else 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load model from checkpoint
    data_dir = os.path.expandvars(config["general"]["save_path"])
    if agent.load_pretrained:
        print("Checking {} for {}".format(data_dir, agent.load_from_tag))
        if os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    training_method = config["general"]["training_method"]
    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    repeats = config["general"]["evaluate"]["repeats"]

    # iterate through all environments
    for eval_env_type in eval_envs:
        # iterate through all controllers
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
            print("Setting controller: %s" % controller_type)
            # iterate through all splits
            for eval_path in eval_paths:
                print("Evaluating: %s" % eval_path)
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type
                experiment_name = config["general"]["evaluate"]["eval_experiment_tag"]

                # change output folder to eval 
                eval_name = experiment_name + "_00" + str(num_files_in_save_output_dir)
                output_folder = os.path.join(save_output_dir, eval_name)
                os.makedirs(output_folder, exist_ok=True)

                # setting text logger and logging level
                text_logger = logging.getLogger("alfworld.agents.agent.text_dagger_agent")
                text_logger.setLevel(logging.INFO)
                logger_path = os.path.join(output_folder, "text_dagger_agent.log")
                file_handler = logging.FileHandler(logger_path)
                file_handler.setLevel(logging.INFO)
                fromatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                text_logger.addHandler(file_handler)

                print(f'config type: {config["general"]["evaluate"]["env"]["type"]}')
                alfred_env = get_environment(config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
                eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)

                # evaluate method
                if training_method == "dagger":
                    results = evaluate_dagger(eval_env, agent, alfred_env.num_games*repeats)
                elif training_method == "dqn":
                    results = evaluate_dqn(eval_env, agent, alfred_env.num_games*repeats)
                else:
                    raise NotImplementedError()

                # save results to json
                split_name = eval_path.split("/")[-1]
                # experiment_name = config["general"]["evaluate"]["eval_experiment_tag"]

                # change output folder to store json results
                results_json = os.path.join(output_folder, "{}_{}_{}_{}.json".format(experiment_name, eval_env_type.lower(), controller_type, split_name))

                with open(results_json, 'w') as f:
                    json.dump(results, f, indent=4, sort_keys=True)
                print("Saved %s" % results_json)

                eval_env.close()


if __name__ == '__main__':
    run_eval()
