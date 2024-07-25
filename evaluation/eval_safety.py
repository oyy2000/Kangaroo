"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import torch
import numpy as np
import shortuuid

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm


def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        temperature = 0.0,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []

    # hidden_states ------
    output_s = []
    output_ids_s = []
    all_hidden_states_s = []
    # hidden_states ------
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
                temperature,
                **kwargs,
            )
        )
        # output, output_ids, all_hidden_states = get_answers_func(
        #         model,
        #         tokenizer,
        #         forward_func,
        #         model_id,
        #         questions[i: i + chunk_size],
        #         answer_file,
        #         max_new_tokens,
        #         num_choices,
        #         temperature,
        #         **kwargs,
        #     )
        # output_s.append(output)
        # output_ids_s.append(output_ids)
        # all_hidden_states_s.append(all_hidden_states)

    if use_ray:
        ray.get(ans_handles)
    
    # return output_s, output_ids_s, all_hidden_states_s

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
        temperature,
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    accept_lengths_tree = []
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
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
    
            try:
                torch.cuda.synchronize()
                start_time = time.time()

                if model_id == 'vicuna-7b-v1.3-vanilla-float16-temp-0.0':
                    output_ids, new_token, idx, accept_length_tree, _ = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        temperature,
                        **kwargs,
                    )
                elif model_id == 'kangaroo_casestudy':
                    output_ids, new_token, idx, accept_length_tree, all_hidden_states = forward_func(
                        inputs,
                        model,
                        tokenizer,
                        max_new_tokens,
                        **kwargs,
                    )
                else:
                    output_ids, new_token, idx, accept_length_tree, _ = forward_func(
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
                
                midlayer_predictions = []
                for token_id in range(len(all_hidden_states)):
                    token_prediction = []
                    token_hidden_states = torch.stack(all_hidden_states[token_id], dim=1)
                    logits = model.head_model(token_hidden_states).float() # batchsize, vocab_size
                    output_token = torch.argmax(logits[:, :, :], dim=-1)
                    midlayer_predictions.append(output_token[0].cpu().tolist())


                transposed_midlayer_predictions = list(zip(*midlayer_predictions))
                transposed_midlayer_predictions = [list(row) for row in transposed_midlayer_predictions]
                
                midlayer_predictions = transposed_midlayer_predictions
                # import ipdb
                # ipdb.set_trace()
                # output_ids : LLM 输出的所有token的 vocab id     -- midlayer_prediction : LLM所有输出token 对应的2-32层分别的预测 vocab id
            except RuntimeError as e:
                print("ERROR when forwarding question ID: ", question["question_id"])
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
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            cur_accept_lengths_tree.extend(accept_length_tree)
            conv.messages[-1][-1] = output
        
        # torch.cuda.empty_cache()
        choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time,
                        "accept_lengths": cur_accept_lengths_tree})

        try:
            midlayer_tokens = {}
            for i in range(len(midlayer_predictions)):
                # Convert each prediction (which is a list of token IDs) to tokens.
                tokens = tokenizer.convert_ids_to_tokens(midlayer_predictions[i])
                midlayer_tokens[i] = tokens
            midlayer_setences = {}
            for i in range(len(midlayer_predictions)):
                midlayer_setences[i] = tokenizer.decode(midlayer_predictions[i], spaces_between_special_tokens=False)
    
        except RuntimeError as e:
            print("ERROR when decoding midlayer predictions")
            midlayer_tokens = "ERROR"

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "label": question["label"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
                "res": output,
                "prompt": q_prompt,
                "source": question["source"],
                "midlayer_predictions": midlayer_predictions,
                # "midlayer_predictions_len": len(midlayer_predictions),
                "midlayer_tokens": midlayer_tokens,
                "midlayer_setences": midlayer_setences,
            }
            fout.write(json.dumps(ans_json) + "\n")
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    return output, output_ids, all_hidden_states


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])



# import os
# import time
# import json
# import shortuuid
# import numpy as np
# from tqdm import tqdm
# import torch

# @torch.inference_mode()
# def get_model_answers(
#         model,
#         tokenizer,
#         forward_func,
#         model_id,
#         questions,
#         answer_file,
#         max_new_tokens,
#         num_choices,
#         **kwargs,
# ):
#     model.eval()
#     print('Check model training state:', model.training)

#     cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
#     print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

#     # Set checkpoint file path in the same directory as answer_file
#     checkpoint_file = os.path.join(os.path.dirname(answer_file), 'checkpoint.json')

#     # Load checkpoints if exists
#     if os.path.exists(checkpoint_file):
#         with open(checkpoint_file, 'r') as f:
#             checkpoints = json.load(f)
#     else:
#         checkpoints = {}

#     accept_lengths_tree = []
#     all_answers = []

#     # Load existing answers if the file exists
#     if os.path.exists(answer_file):
#         with open(answer_file, 'r') as f:
#             for line in f:
#                 all_answers.append(json.loads(line))

#     for question in tqdm(questions):
#         question_id = question["question_id"]
#         if question_id in checkpoints:
#             print(f"Skipping question {question_id} as it is already processed.")
#             continue

#         choices = []
#         for i in range(num_choices):
#             cur_accept_lengths_tree = []
#             conv = get_conversation_template("vicuna")
#             turns = []
#             idxs = []
#             new_tokens = []
#             wall_time = []
#             q_prompt = question["prompt"]
#             conv.append_message(conv.roles[0], q_prompt)
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()
#             inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
#             input_ids = inputs.input_ids
#             try:
#                 torch.cuda.synchronize()
#                 start_time = time.time()
#                 output_ids, new_token, idx, accept_length_tree = forward_func(
#                     inputs,
#                     model,
#                     tokenizer,
#                     max_new_tokens,
#                     **kwargs,
#                 )
#                 torch.cuda.synchronize()
#                 total_time = time.time() - start_time
#                 accept_lengths_tree.extend(accept_length_tree)
#                 output_ids = output_ids[0][len(input_ids[0]):]
#             except RuntimeError as e:
#                 print("ERROR when forwarding question ID: ", question_id)
#                 output = "ERROR"

#             try:
#                 if conv.stop_token_ids:
#                     stop_token_ids_index = [
#                         i
#                         for i, id in enumerate(output_ids)
#                         if id in conv.stop_token_ids
#                     ]
#                     if len(stop_token_ids_index) > 0:
#                         output_ids = output_ids[: stop_token_ids_index[0]]

#                 output = tokenizer.decode(
#                     output_ids,
#                     spaces_between_special_tokens=False,
#                 )
#                 if conv.stop_str and output.find(conv.stop_str) > 0:
#                     output = output[: output.find(conv.stop_str)]
#                 for special_token in tokenizer.special_tokens_map.values():
#                     if isinstance(special_token, list):
#                         for special_tok in special_token:
#                             output = output.replace(special_tok, "")
#                     else:
#                         output = output.replace(special_token, "")
#                 output = output.strip()

#                 if conv.name == "xgen" and output.startswith("Assistant:"):
#                     output = output.replace("Assistant:", "", 1).strip()
#             except RuntimeError as e:
#                 print("ERROR question ID: ", question_id)
#                 output = "ERROR"

#             turns.append(output)
#             idxs.append(int(idx))
#             new_tokens.append(int(new_token))
#             wall_time.append(total_time)
#             cur_accept_lengths_tree.extend(accept_length_tree)
#             conv.messages[-1][-1] = output
#             choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time,
#                             "accept_lengths": cur_accept_lengths_tree})

#         ans_json = {
#             "question_id": question_id,
#             "label": question["label"],
#             "answer_id": shortuuid.uuid(),
#             "model_id": model_id,
#             "choices": choices,
#             "tstamp": time.time(),
#             "res": output,
#             "prompt": q_prompt,
#             "source": question["source"]
#         }
#         all_answers.append(ans_json)

#         checkpoints[question_id] = True
#         with open(checkpoint_file, 'w') as f:
#             json.dump(checkpoints, f)

#     # Write all answers back to the file
#     os.makedirs(os.path.dirname(answer_file), exist_ok=True)
#     with open(answer_file, 'w') as fout:
#         for ans in all_answers:
#             fout.write(json.dumps(ans) + "\n")

#     print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
