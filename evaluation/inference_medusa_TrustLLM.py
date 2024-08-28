"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""

from medusa.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
import argparse
import torch
import json
import os
import time
import numpy as np
import shortuuid
import torch.nn.functional as F
from trustllm.task import safety
from trustllm.utils import file_process

from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from kangaroo.kangaroo_model import KangarooModel
from fastchat.model import get_conversation_template
from tqdm import tqdm
import random
from transformers import set_seed

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)

# For reproducibility in convolution operations, etc.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@contextmanager
def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def generation(
        model,
        tokenizer,
        model_id,
        question_file,
        answer_file_dir,
        answer_file_name,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    with open(question_file) as f:
        original_data = json.load(f)
    answer_file = os.path.join(answer_file_dir, answer_file_name)
    if os.path.exists(answer_file):
        with open(answer_file, 'r') as f:
            res_data = json.load(f)
    else:
        res_data = original_data

    questions = res_data

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    for i in range(0, len(questions), chunk_size):
        get_model_answers(
            model,
            tokenizer,
            model_id,
            questions[i: i + chunk_size],
            answer_file,
            max_new_tokens,
            num_choices,
            **kwargs,
        )

@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    accept_lengths_tree = []
    wall_time_list = []
    for question in tqdm(questions):
        # Dump answers
        if "res" not in question or not question['res']:
            choices = []
            # torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            q_prompt = question["prompt"]
            conv.append_message(conv.roles[0], q_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            total_time = 0
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx, wall_times = medusa_forward(
                    input_ids,
                    model,
                    tokenizer,
                    **kwargs,
                )
                
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
            except RuntimeError as e:
                print("ERROR when forwarding question: ", question["prompt"])
                output = "ERROR"

            try:
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question: ", question["prompt"])
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))

            conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            # choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time,
            #                 "accept_lengths": cur_accept_lengths_tree})
            question['res'] = output
            save_json(questions, answer_file)
        # else:
            # print("Skip question: ", question["prompt"])
            # accept_lengths_tree.extend(question['accept_lengths'])
            # compute the average wall time of all questions
    
    # print("#Mean wall time: ", wall_times)
    # print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    # create a log file if not exists
    log_file_path = os.path.join(answer_file_dir, f"log_{args.subtask}.txt")

    
        
    with open(log_file_path, 'w') as f:
        evaluator = safety.SafetyEval()
        jailbreak_data = file_process.load_json(answer_file)
        jailbreak_score = evaluator.jailbreak_eval(jailbreak_data, eval_type='total')
        print(f"jailbreak score: {jailbreak_score}\n")
        f.write(f"Model: {args.model_id}\n")
        f.write(f"Total questions: {len(questions)}\n")
        f.write(f"Mean wall time: {np.mean(wall_time_list)}\n")
        f.write(f"Mean accepted tokens: {np.mean(accept_lengths_tree)}\n")  
        f.write(f"jailbreak score: {jailbreak_score}\n")


def medusa_forward(input_ids, model, tokenizer, max_steps = 512, **kwargs):
    max_steps = 256
    wall_times = {'medusa': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}
    
    with timed(wall_times, 'init'):
        if hasattr(model, "medusa_choices") and model.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = model.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=model.base_model.device
            )
        model.medusa_buffers = medusa_buffers
        model.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_medusa_mode(model)
        medusa_logits, logits = initialize_medusa(
                input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
        )
    new_token = 0

    for idx in range(max_steps): 
        with timed(wall_times, 'medusa'):
            candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                )

        with timed(wall_times, 'tree'):
            medusa_logits, logits, outputs = tree_decoding(
                    model,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )

        with timed(wall_times, 'posterior'):
            best_candidate, accept_length = evaluate_posterior(
                    logits, candidates, temperature, epsilon, delta
                )
        
        with timed(wall_times, 'update'):
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    medusa_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break

    return input_ids, new_token, idx, wall_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
    )
    parser.add_argument("--model-id", type=str)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="The threshold for fallback.",
    )

    parser.add_argument(
        "--exitlayer",
        type=int,
        default=2,
        help="The exitlayer.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="The number of GPUs per model.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    parser.add_argument(
        "--subtask",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--do_sample",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hyper_k",
        type=int,
    )

    parser.add_argument(
        "--hyper_p",
        type=float,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.09
    )

    args = parser.parse_args()
    
    question_file = f"data/eval_data/{args.task}/{args.subtask}.json"

    model_name = 'checkpoints/medusa-vicuna-7b-v1.3'
    temperature = args.temperature
    epsilon = args.epsilon
    delta = args.delta

    model = MedusaModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    
    if args.do_sample == "top_k":
        parameter = f"{args.hyper_k}"
    elif args.do_sample == "top_p":
        parameter = f"{args.hyper_p}"
    else:
        parameter = f"epsilon_{args.epsilon}_delta_{args.delta}"        

    model_id = f"medusa-vicuna-7b-v1.3-{args.do_sample}_{parameter}_temp_{args.temperature}"
    args.model_id = model_id
    answer_file_dir = f"data/{args.bench_name}/{args.model_id}/{args.task}"
    os.makedirs(answer_file_dir, exist_ok=True)
    answer_file_name = f"{args.subtask}.json"
    medusa_choices = mc_sim_7b_63
    print(f"Output to {answer_file_dir}/{answer_file_name}")

    generation(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model_id,
        question_file=question_file,
        answer_file_dir=answer_file_dir,
        answer_file_name=answer_file_name,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        do_sample=args.do_sample,
        temperature=args.temperature,
        hyper_k=args.hyper_k,
        hyper_p=args.hyper_p,
        epsilon=args.epsilon,
        delta=args.delta,
        SPECULATIVE_DECODING_STEPS=args.steps,
        EARLY_STOP_LAYER=args.exitlayer,
        medusa_choices=medusa_choices,
    )
