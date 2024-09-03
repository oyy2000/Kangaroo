import subprocess
from itertools import product
from multiprocessing import Queue
import time
import argparse
import threading
import psutil

def parse_arguments():
    parser = argparse.ArgumentParser(description="Task Executor for GPU-based Evaluation")
    parser.add_argument("--bench_type", type=str, required=True, help="Benchmark type (e.g., MMLU or TrustLLM)")
    parser.add_argument("--model_type", type=str, required=True, help="Model type (e.g., Kangaroo or Baseline)")
    parser.add_argument("--GPU_number", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--interval", type=int, default=20, help="Interval between command executions in seconds")
    parser.add_argument("--ramp_up_time", type=int, default=120, help="Time to wait for GPU memory to ramp up (seconds)")
    return parser.parse_args()

def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    memory_usage = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
    return memory_usage

def get_free_gpus(threshold=10000):
    memory_usage = get_gpu_memory_usage()
    return [i for i, mem in enumerate(memory_usage) if mem < threshold]

def run_command(gpu_id, command):
    print(f"Running on GPU {gpu_id}: {command}")
    process = subprocess.Popen(command, shell=True)
    return process

def monitor_gpu_usage(gpu_id, target_memory, timeout):
    start_time = time.time()
    while time.time() - start_time < timeout:
        memory_usage = get_gpu_memory_usage()[gpu_id]
        if memory_usage >= target_memory:
            return True
        time.sleep(5)
    return False
def worker(task_queue, args, gpu_status):
    while True:
        task = task_queue.get()
        if task is None:
            break

        index, params = task
        while True:
            # Combine both approaches
            currently_free_gpus = set(get_free_gpus(threshold=6000))  # Adjusted threshold
            status_free_gpus = set([gpu for gpu, status in gpu_status.items() if status == 'free'])
            
            # Only consider GPUs that are free according to both criteria
            free_gpus = list(currently_free_gpus.intersection(status_free_gpus))

            if free_gpus:
                gpu_id = free_gpus[0]
                gpu_status[gpu_id] = 'busy'
                command = build_command(args, gpu_id, params)
                process = run_command(gpu_id, command)

                # Monitor GPU memory usage
                if monitor_gpu_usage(gpu_id, 6000, args.ramp_up_time):
                    print(f"Task {index} on GPU {gpu_id} has ramped up. Waiting for completion...")
                    process.wait()
                else:
                    print(f"Task {index} on GPU {gpu_id} failed to ramp up. Terminating...")
                    process.terminate()

                gpu_status[gpu_id] = 'free'
                print(f"Task {index} completed. Waiting for {args.interval} seconds before next task...")
                time.sleep(args.interval)
                break
            else:
                print(f"No GPU available for task {index}. Waiting...")
                time.sleep(5)

def build_command(args, gpu_id, params):
    if args.model_type == "Kangaroo":
        top_p, temperature = params
        if args.bench_type == "MTBench":
            return (
                f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_kangaroo_MTBench '
                f'--model-path "vicuna-7b-v1.3" --adapter-path "kangaroo-vicuna-7b-v1.3" '
                f'--exitlayer 2 --model-id "kangaroo-vicuna-7b-v1.3" --threshold 0.6 '
                f'--temperature {temperature} --steps 6 --bench-name "mt_bench" '
                f'--dtype "float16" --max-new-token 256 --max-length 512 '
                f'--do_sample "top_p" --hyper_p {top_p}'
            )
        elif args.bench_type == "TrustLLM":
            return (
                f"CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_kangaroo_typical_sampling "
                f"--task 'safety' --subtask 'jailbreak' --adapter-path 'kangaroo-vicuna-7b-v1.3' "
                f"--exitlayer 2 --model-path 'vicuna-7b-v1.3' --threshold 0.6 --temperature {temperature} "
                f"--steps 6 --bench-name '{args.bench_type}' --dtype 'float16' --do_sample 'top_p' "
                f"--max-new-tokens 1024 --hyper_p {top_p}"
            )
    elif args.model_type == "Fusion":
        temperature, fusion_layer = params
        if args.bench_type == "TrustLLM":
            return (
                f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_distribution_fusion '
                f'--task "safety" --subtask "jailbreak" --model-path "vicuna-7b-v1.3" --adapter-path "kangaroo-vicuna-7b-v1.3" '
                f'--exitlayer 2 --model-id "kangaroo-vicuna-7b-v1.3" --threshold 0.6 '
                f'--temperature {temperature} --steps 6 --bench-name "TrustLLM" '
                f'--dtype "float16" --max-new-token 256 --fusion-layer {fusion_layer} '
            )
    elif args.model_type in ["Baseline", "Medusa"]:
        temperature = params[0]
        if args.bench_type == "MTBench":
            return (
                f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_{args.model_type.lower()}_{args.bench_type} '
                f'--model-path "{args.model_type.lower()}-vicuna-7b-v1.3" '
                f'--model-id "{args.model_type.lower()}-vicuna-7b-v1.3" --temperature {temperature} '
                f'--bench-type "mt_bench" --max-new-token 256'
            )

            """
            CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference_baseline_MTBench --model-path "baseline-vicuna-7b-v1.3" --model-id "baseline-vicuna-7b-v1.3" --temperature 1.1 --bench-name "mt_bench" --max-new-token 256
            """

        elif args.bench_type == "TrustLLM":
            return (
                f'CUDA_VISIBLE_DEVICES={gpu_id} python -m evaluation.inference_{args.model_type.lower()}_{args.bench_type} '
                f'--task "safety" --subtask "jailbreak" --model-path "vicuna-7b-v1.3" '
                f'--model-id "{args.model_type.lower()}-vicuna-7b-v1.3" --threshold 0.6 --temperature {temperature} '
                f'--steps 6 --bench-type {args.bench_type} --dtype "float16" --do_sample "top_p" '
                f'--max-new-tokens 1024 --hyper_p 0.5'
            )
            """
             CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference_baseline_TrustLLM --task "safety" --subtask "jailbreak" --model-path "vicuna-7b-v1.3" --model-id "baseline-vicuna-7b-v1.3" --threshold 0.6 --temperature 1.1 --steps 6 --bench-type "TrustLLM" --dtype "float16" --do_sample "top_p" --max-new-tokens 1024 --hyper_p 0.5
            """
def main():
    args = parse_arguments()
    
    # temperature_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
    temperature_values = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
    
    # temperature_values = [0.0]
    
    # temperature_values = [0.0]
    top_p_values = [0.3, 0.5]
    fusion_layers = [5, 10, 15, 25]
    alphas = [0.1, 0.01, 0.001]
    
    task_queue = Queue()
    
    if args.model_type == "Kangaroo":
        combinations = list(product(top_p_values, temperature_values))
        for i, combo in enumerate(combinations):
            task_queue.put((i, combo))
    elif args.model_type == "Fusion":
        combinations = list(product(temperature_values, fusion_layers, alphas))
        for i, combo in enumerate(combinations):
            task_queue.put((i, combo))
    else:    
        for i, temp in enumerate(temperature_values):
            task_queue.put((i, (temp,)))
    
    # Add sentinel values to signal workers to exit
    for _ in range(args.GPU_number):
        task_queue.put(None)
    
    gpu_status = {i: 'free' for i in range(args.GPU_number)}
    
    workers = []
    for _ in range(args.GPU_number):
        t = threading.Thread(target=worker, args=(task_queue, args, gpu_status))
        t.start()
        workers.append(t)
    
    for t in workers:
        t.join()

if __name__ == "__main__":
    main()