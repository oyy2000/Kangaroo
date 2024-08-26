# from trustllm.task import safety
# from trustllm.utils import file_process
# from trustllm import config

# evaluator = safety.SafetyEval()

# jailbreak_data = file_process.load_json('results/kangaroo/temper-0.json')
# print(evaluator.jailbreak_eval(jailbreak_data, eval_type='total')) # return overall RtA
# print(evaluator.jailbreak_eval(jailbreak_data, eval_type='single')) # return RtA dict for each kind of jailbreak ways


## -------------

from trustllm.task.pipeline import run_safety
import json

# def load_questions(question_file: str):
#     """Load questions from a file."""
#     questions = []
#     with open(question_file, "r") as ques_file:
#         for line in ques_file:
#             if line:
#                 questions.append(json.loads(line))
#     return questions

# file = load_questions("results/kangaroo/temper-0.json")

# with open('results/kangaroo/temper-0_list.json','w+') as f:
#     f.write(json.dumps(file))

safety_results = run_safety(  
    jailbreak_path="results/kangaroo/temper-0_list.json"
)


## -------------
# from trustllm.dataset_download import download_dataset

# download_dataset(save_path='./trustllm_datasets')