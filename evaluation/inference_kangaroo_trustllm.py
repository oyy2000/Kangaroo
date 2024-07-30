"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import torch
import json
import os
import time
import numpy as np
import shortuuid

from transformers import AutoModelForCausalLM, AutoTokenizer
from kangaroo.kangaroo_model import KangarooModel
from fastchat.model import get_conversation_template
from tqdm import tqdm


def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask


def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask
    


def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold=0.3, posterior_alpha = 0.09, top_p=0.8, sampling = 'typical', fast = True
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
        
    if sampling == 'typical':
        if fast:
            posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_length
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha, fast)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()
        
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate, accept_length
    
    if sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def generation(
        model,
        tokenizer,
        forward_func,
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

    get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):

        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )



@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
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
            cur_accept_lengths_tree = []
            # torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
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
                output_ids, new_token, idx, accept_length_tree = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                accept_lengths_tree.extend(accept_length_tree)
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
            wall_time.append(total_time)
            cur_accept_lengths_tree.extend(accept_length_tree)
            conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            # choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time,
            #                 "accept_lengths": cur_accept_lengths_tree})
            question['res'] = output
            question['accept_lengths'] = cur_accept_lengths_tree
            question['wall_time'] = total_time
            save_json(questions, answer_file)
        else:
            accept_lengths_tree.extend(question['accept_lengths'])
            # compute the average wall time of all questions
        wall_time_list.append(total_time)
    
    print("#Mean wall time: ", np.mean(wall_time_list))
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    # create a log file if not exists
    log_file_path = os.path.join(answer_file_dir, f"log_{args.subtask}.txt")

    with open(log_file_path, 'w') as f:
        f.write(f"Model: {args.model_id}\n")
        f.write(f"Total questions: {len(questions)}\n")
        f.write(f"Mean wall time: {np.mean(wall_time_list)}\n")
        f.write(f"Mean accepted tokens: {np.mean(accept_lengths_tree)}\n")    


def kangaroo_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, max_length = 2048, EARLY_STOP_LAYER = 2, SPECULATIVE_DECODING_STEPS = 6, threshold = 0.6):
    context_tokens = inputs.input_ids
    device = context_tokens.device
    token_eos = tokenizer.eos_token_id
    batch_size, context_length = context_tokens.shape
    global_tokens = torch.ones((batch_size, max_length), dtype=torch.long, device=device) * token_eos
    global_position_ids = torch.LongTensor([[i for i in range(max_length)]]).to(device)
    accept_length_list = [1]

    start_index = context_length
    global_tokens[:, :start_index] = context_tokens

    # Init KV-chache and sample the first token
    with torch.no_grad():
        position_ids = global_position_ids[:, :start_index]
        output = model.base_model(context_tokens, position_ids=position_ids, output_hidden_states=True)
        model.base_model.past_key_values = list(output.past_key_values)
        hidden_state = output.hidden_states[-1]
        logits = output.logits # batchsize, input_length, vocab_size
        global_tokens[:, start_index] = torch.argmax(logits[:, -1, :], dim=-1).item()
        hidden_state_early = output.hidden_states[EARLY_STOP_LAYER]

        # KV-cache for the adapter
        hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(inputs_embeds=hidden_state_early[:,:,:], position_ids=global_position_ids[:, :context_length], use_cache=True) 

    total_inference_steps = 0

    with torch.no_grad():
        max_infer_steps = min(max_length, start_index + max_new_tokens)
        stop = False

        while start_index < max_infer_steps - 1 - SPECULATIVE_DECODING_STEPS:

            start_index_copy = start_index
            end_index = start_index + 1
            
            # STEP 1: Small model decoding
            for step in range(1 + SPECULATIVE_DECODING_STEPS):
                assert adapter_past_key_values[0][0].shape[2] <= end_index-1, "{} - {}".format(adapter_past_key_values[0][0].shape, end_index-1)
                in_tokens_small = global_tokens[:, end_index-1:end_index]
                if adapter_past_key_values[0][0].shape[2] < end_index-1: 
                    # As illustrated in the framework of Kangaroo, once all drafted tokens are accepted, the KV-cache of the last draft token for the adapter is missing.
                    position_ids = global_position_ids[:, start_index-1:end_index]
                    hidden_state_early_last = exited_hidden_states[:,-1:,:]
                else:
                    position_ids = global_position_ids[:, end_index-1:end_index]
                    hidden_state_early_last = None
                
                hidden_state_early = model.base_model.forward_draft_or_large_model(in_tokens_small=in_tokens_small[:,-1:], position_ids=position_ids[:,-1:])
                
                if step==0:
                    exited_hidden_states = None

                exited_hidden_states = hidden_state_early if exited_hidden_states is None else torch.cat([exited_hidden_states, hidden_state_early], dim = 1)
                
                if hidden_state_early_last is not None:
                    hidden_state_early = torch.cat([hidden_state_early_last, hidden_state_early], dim = 1)

                # early exiting 
                if step == SPECULATIVE_DECODING_STEPS or (step > 0 and predict_score < threshold):
                    break

                hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(inputs_embeds=hidden_state_early, position_ids=position_ids, past_key_values=adapter_past_key_values, use_cache=True)

                predict_logits = model.head_model(hidden_state[:,-1:,:]).float() 
                global_tokens[:, end_index] = torch.argmax(predict_logits[:, -1, :], dim=-1)
                
                end_index += 1
                predict_score = predict_logits.softmax(dim=-1).max().item()

            # STEP2: Big model inference
            position_ids = global_position_ids[:, start_index:end_index]
            assert model.base_model.past_key_values[EARLY_STOP_LAYER][0].shape[2] == start_index, "{} - {}".format(model.base_model.past_key_values[EARLY_STOP_LAYER][0].shape, start_index)
            assert exited_hidden_states.shape[1] == position_ids.shape[1]
            hidden_state_, hidden_state = model.base_model.forward_draft_or_large_model(in_features_large=exited_hidden_states, position_ids=position_ids)
            
            logits = model.head_model(hidden_state).float() # batchsize, input_length, vocab_size
            output_length = end_index - start_index
            if do_sample:
                # Generate top-k candidates
                top_k = 10  # Define the number of top candidates to consider
                top_probs, top_indices = torch.topk(logits, top_k, dim=-1)
                candidates = top_indices.squeeze(0)  # Remove batch dimension

                # Evaluate posterior probabilities and choose the best candidate
                best_candidate, accept_length = evaluate_posterior(
                    logits.squeeze(0),  # Remove batch dimension
                    candidates,
                    temperature,
                    posterior_threshold=posterior_threshold,
                    posterior_alpha=posterior_alpha,
                    top_p=top_p,
                    sampling=sampling
                )
                global_tokens[:, end_index] = best_candidate
            else:
            
            # output_tokens = torch.argmax(logits[:, :, :], dim=-1)

            # # Verification for greedy decoding
            # output_lenght = end_index - start_index
            # for i in range(output_lenght):
            #     if i == output_lenght-1 or output_tokens[0, i] == token_eos or output_tokens[0, i] != global_tokens[0, start_index+1+i]:
            #         global_tokens[0, start_index+1+i] = output_tokens[0, i]
            #         start_index = start_index+1+i
            #         if output_tokens[0, i] == token_eos:
            #             stop = True
            #         break

            accept_length_list.append(start_index - start_index_copy)
            hidden_state = hidden_state[:, :output_length-(end_index-start_index), :]

            # STEP 4: Post process KV-cache
            if model.base_model.past_key_values[0][0].shape[2] > start_index:
                past_key_values_large_ = []
                for k,v in model.base_model.past_key_values:
                    past_key_values_large_.append((k[:,:,:start_index,:], v[:,:,:start_index,:]))
                model.base_model.past_key_values = past_key_values_large_

            if adapter_past_key_values[0][0].shape[2] > start_index:
                adapter_past_key_values_ = []
                for k,v in adapter_past_key_values:
                    adapter_past_key_values_.append((k[:,:,:start_index,:], v[:,:,:start_index,:]))
                adapter_past_key_values = tuple(adapter_past_key_values_)
                del adapter_past_key_values_
            
            total_inference_steps += 1

            if stop:
                break

    output_ids = global_tokens[0, :start_index+1].tolist()
    new_token = start_index - context_length + 1
    idx = len(accept_length_list) - 1
    return [output_ids], new_token, idx, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
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
        "--threshold",
        type=float,
        default=0.4,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--exitlayer",
        type=int,
        default=2,
        help="The temperature for medusa sampling.",
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

    args = parser.parse_args()
    
    question_file = f"data/eval_data/{args.task}/{args.subtask}.json"

    model = KangarooModel(args.model_path, args.adapter_path, args, EARLY_STOP_LAYER = args.exitlayer)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    do_sample = True

    answer_file_dir = f"data/{args.bench_name}/{args.model_id}/{args.task}"
    os.makedirs(answer_file_dir, exist_ok=True)
    answer_file_name = f"{args.subtask}.json"
    
    print(f"Output to {answer_file_dir}/{answer_file_name}")

    generation(
        model=model,
        tokenizer=tokenizer,
        forward_func=kangaroo_forward,
        model_id=args.model_id,
        question_file=question_file,
        answer_file_dir=answer_file_dir,
        answer_file_name=answer_file_name,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        do_sample=do_sample,
        threshold=args.threshold,
        SPECULATIVE_DECODING_STEPS=args.steps,
        EARLY_STOP_LAYER=args.exitlayer
    )
