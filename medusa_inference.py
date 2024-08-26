import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # define GPU id, remove if you want to use all GPUs available
import torch
from tqdm import tqdm
import time
from contextlib import contextmanager
import numpy as np
from medusa.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
import transformers
from huggingface_hub import hf_hub_download

model_name = 'checkpoints/medusa-vicuna-7b-v1.3'

@contextmanager
def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

def medusa_forward(input_ids, model, tokenizer, medusa_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 512):
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
                    logits, candidates, temperature, posterior_threshold, posterior_alpha
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


model = MedusaModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
tokenizer = model.get_tokenizer()

medusa_choices = mc_sim_7b_63

temperature = 0.
posterior_threshold = 0.09
posterior_alpha = 0.3

prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"

def prompt_inference(prompt):
    
    with torch.inference_mode():
        input_ids = tokenizer([prompt]).input_ids
        output_ids, new_token, idx, wall_time = medusa_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        medusa_choices,
                        temperature,
                        posterior_threshold,
                        posterior_alpha,
                    )
        output_ids = output_ids[0][len(input_ids[0]) :]
        print("Output length:", output_ids.size(-1))
        print("Compression ratio:", new_token / idx)

    output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
    return output

print(prompt_inference(prompt))

import ipdb
ipdb.set_trace()

A = [1,2,3]

# analyzing walltime
# max_length = 50

# def format_string(text, value, max_length):
#     value_str = "{:.3f}".format(value)
#     return f"{text:<{max_length - len(value_str)}}{value_str}"

# time_init = np.sum(wall_time['init'] )
# time_medusa = np.sum(wall_time['medusa'] )
# time_tree = np.sum(wall_time['tree'] )
# time_posterior = np.sum(wall_time['posterior'] )
# time_update = np.sum(wall_time['update'] )
# time_total = time_init + time_medusa + time_tree + time_posterior + time_update

# print('='*max_length)
# print(format_string("Wall time init: ", time_init, max_length))
# print(format_string("Wall time medusa: ", time_medusa, max_length))
# print(format_string("Wall time Tree: ", time_tree, max_length))
# print(format_string("Wall time Posterior: ", time_posterior, max_length))
# print(format_string("Wall time Update: ", time_update, max_length))
# print('-'*max_length)
# print(format_string("Wall time portion medusa: ", time_medusa / time_total, max_length))
# print(format_string("Wall time portion Tree: ", time_tree / time_total, max_length))
# print(format_string("Wall time portion Posterior: ", time_posterior / time_total, max_length))
# print(format_string("Wall time portion Update: ", time_update / time_total, max_length))
# print('-'*max_length)
# print(format_string("Tokens/second: ", new_token / time_total, max_length))
# print('='*max_length)

