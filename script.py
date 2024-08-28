from itertools import product

# Define ranges for top_p and temperature
top_p_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# temperature_values = [0.1, 0.3, 0.5, 0.7, 0.9]
temperature_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# temperature_values.append([3, 5, 10, 20, 50])  

# Create a list of all combinations
combinations = list(product(top_p_values, temperature_values))

# Number of GPUs available
num_gpus = 8  # Adjust this to match the number of GPUs you have


# Function to run a single command
# def run_command(index, top_p, temperature):
#     gpu_id = index % num_gpus  # Round-robin assignment of GPU ID
#     command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_kangaroo_typical_sampling --task 'safety' --subtask 'jailbreak' --adapter-path 'kangaroo-vicuna-7b-v1.3' --exitlayer 2 --model-path 'vicuna-7b-v1.3' --threshold 0.6 --temperature {temperature} --steps 6 --model-id 'vicuna-7b-v1.3-kangaroo-top_p_{top_p}_temp_{temperature}' --bench-name 'Kangaroo' --dtype 'float16' --do_sample 'top_p' --max-new-tokens 1024 --hyper_p {top_p}"
#     print(f"Running on GPU {gpu_id}: {command}")
#     subprocess.call(command, shell=True)

# # Prepare arguments for pool.starmap
# args = [(i, top_p, temperature) for i, (top_p, temperature) in enumerate(combinations)]

# # Run all tasks in parallel
# with Pool() as pool:
#     pool.starmap(run_command, args)
    
import os
import subprocess
from itertools import product
from multiprocessing import Pool

# Define ranges for top_p and temperature
top_p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
temperature_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Create a list of all combinations
combinations = list(product(top_p_values, temperature_values))

# Number of GPUs available
num_gpus = 8  # Adjust this to match the number of GPUs you have

def get_free_gpus():
    # Run nvidia-smi command and get the output
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    # Decode the output and split by lines
    gpu_memory = result.stdout.decode('utf-8').strip().split('\n')
    # Convert to integer and get the indices of the GPUs sorted by free memory (descending order)
    gpu_memory = [(i, int(mem)) for i, mem in enumerate(gpu_memory)]
    sorted_gpus = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
    # Return the list of GPU indices sorted by free memory
    free_gpus = [gpu[0] for gpu in sorted_gpus]
    return free_gpus

# Function to run a single command
def run_command(index, temperature):
    free_gpus = get_free_gpus()  # Get a list of GPUs sorted by free memory
    print(f"Free GPUs: {free_gpus}")
    gpu_id = free_gpus[index % len(free_gpus)]  # Use round-robin assignment of GPU ID
    command = f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_baseline_MMLU --model-path "vicuna-7b-v1.3" --bench-name "MMLU" --temperature {temperature} --dtype "float16" --data-dir "data/MMLU_data" --save-dir "data/MMLU_results"'
    print(f"Running on GPU {gpu_id}: {command}")
    subprocess.call(command, shell=True)

# Prepare arguments for pool.starmap
args = [(i, temperature) for i, temperature in enumerate(temperature_values)]

# Run all tasks in parallel
with Pool() as pool:
    pool.starmap(run_command, args) 
    