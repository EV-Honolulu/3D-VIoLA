import json
import argparse
import os

def count_success_task(data):
    """
    This function counts the number of successful tasks in the provided data.
    It assumes that 'data' is a dictionary with a 'success' key that contains a list of success statuses.
    """
    print(f'keys in data: {data.keys()}')

    success_task = [i for i, v in enumerate(data["res_points"]) if v == 1.0]
    scuccess_count = len(success_task)

    success_task_name = [data["res_info"][i] for i in success_task]

    return scuccess_count, success_task_name

def eval_success(input_path, type):
    """
    This function reads a JSON file, processes it, and writes the output to another JSON file.
    It can be used to evaluate the success of operations by checking the contents of the JSON.
    """
    if type == 'base':
        eval_path = os.path.join(type)
        output_file = os.path.join(eval_path, 'output.json')

    output_data = {}  # Collect all results here

    for folders in os.listdir(eval_path):
        folder_path = os.path.join(eval_path, folders)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.endswith('.json'):
                continue

            input_file = os.path.join(folder_path, file)
            print(f"Processing file: {input_file}")

            with open(input_file, 'r') as infile:
                data = json.load(infile)

            # Count success tasks
            success_count, success_task = count_success_task(data)

            # Store under folder -> file
            if folders not in output_data:
                output_data[folders] = {}
            output_data[folders] = {
                "average_points": data.get("average_points", None),
                "average_steps": data.get("average_steps", None),
                "total_task": len(data.get("res_info", [])),
                "success_rate": success_count,
                "success_task": success_task
            }

        # Write once to JSON
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)
    
    return output_data

def get_success_task_obs(type, output_data):
    """
    This function extracts the successful tasks descriptions and observation and Final actions from log file
    """
    if type == 'base':
        eval_path = os.path.join(type)
    for folder, results in output_data.items():
        print(f"Folder: {folder}")
        log_path = os.path.join(eval_path, folder, 'tex_dagger_agent.log')
        
        for task in results['success_task']:
            print(f"- {task}")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate success of JSON operations.")
    parser.add_argument('--input', type=str, default='eval', help='Input JSON file path')
    parser.add_argument('--type', type=str, default='base', help='Type of evaluation')
    # parser.add_argument('--output', type=str, default='output.json', help='Output JSON file path')
    args = parser.parse_args()

    output_data = eval_success(args.input, args.type)
    # get_success_task_obs(args.type, output_data)
    print("Evaluation completed successfully. Check output.json for results.")