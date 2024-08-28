import subprocess
from itertools import product
from multiprocessing import Pool, Queue
import time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Task Executor for GPU-based Evaluation")
parser.add_argument("--bench_type", type=str, required=True, help="Benchmark type (e.g., MMLU or TrustLLM)")
parser.add_argument("--model_type", type=str, required=True, help="Model type (e.g., Kangaroo or Baseline)")
parser.add_argument("--GPU_number", type=int, required=True, help="Number of GPUs to use")
parser.add_argument("--interval", type=int, default=20, help="Interval between command executions in seconds")

args = parser.parse_args()

# Define ranges for temperature and top_p values
# temperature_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0]
# top_p_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]

temperature_values = [0.2]

top_p_values = [0.5]

# Set the number of GPUs available
num_gpus = args.GPU_number

# Create a list of all combinations
combinations = list(product(top_p_values, temperature_values))

# Initialize a queue to hold waiting tasks
task_queue = Queue()

def get_free_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    gpu_memory = result.stdout.decode('utf-8').strip().split('\n')
    gpu_memory = [(i, int(mem)) for i, mem in enumerate(gpu_memory)]
    sorted_gpus = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
    # kick out the GPUs with less than 1000MB free memory
    sorted_gpus = [gpu for gpu in sorted_gpus if gpu[1] > 16000]
    free_gpus = [gpu[0] for gpu in sorted_gpus]
    
    return free_gpus

def run_command_kangaroo(index, top_p, temperature):
    while True:
        free_gpus = get_free_gpus()
        if free_gpus:
            gpu_id = free_gpus[index % len(free_gpus)]
            if args.model_type == "Kangaroo":
                if args.bench_type == "MTBench":
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_kangaroo_MTBench --model-path "vicuna-7b-v1.3" '
                        f'--adapter-path "kangaroo-vicuna-7b-v1.3" --exitlayer 2 --model-path "vicuna-7b-v1.3" --model-id "kangaroo-vicuna-7b-v1.3" '
                        f'--threshold 0.6 --temperature {temperature} --steps 6  --bench-name "MTBench" --dtype "float16" --max-new-token 256 ' 
                        f'--max-length 512 --do_sample "top_p" --hyper_p {top_p} '
                    )
                elif args.bench_type == "TrustLLM":
                    command = (
                        f"CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_kangaroo_typical_sampling "
                        f"--task 'safety' --subtask 'jailbreak' --adapter-path 'kangaroo-vicuna-7b-v1.3' "
                        f"--exitlayer 2 --model-path 'vicuna-7b-v1.3' --threshold 0.6 --temperature {temperature} "
                        f"--steps 6 "
                        f"--bench-name '{args.bench_type}' --dtype 'float16' --do_sample 'top_p' --max-new-tokens 1024 --hyper_p {top_p}"
                    ) 
                
            print(f"Running on GPU {gpu_id}: {command}")
            subprocess.Popen(command, shell=True)
            break
        else:
            print(f"No GPU available for task {index}. Waiting...")
            time.sleep(20)

    while not task_queue.empty():
        index, top_p, temperature = task_queue.get()
        run_command_kangaroo(index, top_p, temperature)


def run_command_baseline_or_medusa(index, temperature):
    while True:
        free_gpus = get_free_gpus()
        if free_gpus:
            gpu_id = free_gpus[index % len(free_gpus)]
            if args.model_type == "Baseline":
                if args.bench_type == "MTBench":
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_baseline_{args.bench_type} --model-path "vicuna-7b-v1.3" '
                        f'--model-id "baseline-vicuna-7b-v1.3" --temperature {temperature}  --bench-name "mt_bench" --max-new-token 256'
                    )
                elif args.bench_type == "TrustLLM":
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_baseline_{args.bench_type} --task "safety" --subtask "jailbreak" --model-path "vicuna-7b-v1.3" '
                        f'--model-id "baseline-vicuna-7b-v1.3" --threshold 0.6 --temperature {temperature} --steps 6  --bench-type {args.bench_type} --dtype "float16" --do_sample "top_p" --max-new-tokens 1024 --hyper_p 0.5'
                    )
                    
            elif args.model_type == "Medusa":
                if args.bench_type == "MTBench":
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_medusa_{args.bench_type} --model-path "checkpoints/medusa-vicuna-7b-v1.3" '
                        f'--model-id "medusa-vicuna-7b-v1.3" --temperature {temperature}  --bench-name "mt_bench" '
                    )
                elif args.bench_type == "TrustLLM":
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_medusa_{args.bench_type} '
                        f"--task 'safety' --subtask 'jailbreak' --adapter-path 'kangaroo-vicuna-7b-v1.3' "
                        f"--exitlayer 2 --model-path 'vicuna-7b-v1.3' --threshold 0.6 --temperature {temperature} "
                        f"--bench-name '{args.bench_type}' --dtype 'float16' --do_sample 'top_p' --max-new-tokens 1024 "
                    )
            print(f"Running on GPU {gpu_id}: {command}")
            subprocess.Popen(command, shell=True)
            time.sleep(20)
            break
        else:
            print(f"No GPU available for task {index}. Waiting...")
            time.sleep(20)
    while not task_queue.empty():
        index, temperature = task_queue.get()
        run_command_baseline_or_medusa(index, temperature)

def worker_baseline_or_medusa(index, temperature):
    free_gpus = get_free_gpus()
    if free_gpus:
        run_command_baseline_or_medusa(index, temperature)
    else:
        print(f"Task {index} is being queued due to no available GPU.")
        task_queue.put((index, top_p, temperature))

def worker_kangaroo(index, top_p, temperature):
    free_gpus = get_free_gpus()
    if free_gpus:
        run_command_kangaroo(index, top_p, temperature)
    else:
        print(f"Task {index} is being queued due to no available GPU.")
        task_queue.put((index, top_p, temperature))

        
if __name__ == "__main__":
    if args.model_type == "Kangaroo":
        print("Launching Kangaroo tasks...")
        for i, (top_p, temperature) in enumerate(combinations):
            worker_kangaroo(i, top_p, temperature)
            if i < len(combinations) - 1:  # Avoid sleeping after the last task
                print(f"Waiting for {args.interval} seconds before launching the next task...")
                time.sleep(args.interval)
    elif args.model_type == "Baseline" or args.model_type == "Medusa":
        print("Launching Baseline tasks...")
        for i, temperature in enumerate(temperature_values):
            worker_baseline_or_medusa(i, temperature)
            if i < len(combinations) - 1:
                print(f"Waiting for {args.interval} seconds before launching the next task...")
                time.sleep(args.interval)