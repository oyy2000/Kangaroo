from trustllm.generation.generation import LLMGeneration

llm_gen = LLMGeneration(
    model_path="../Kangaroo/cache/CKPT/vicuna-7b-v1.3", 
    test_type="safety", 
    data_path="trustllm_datasets",
    model_name="vicuna-7b-v1.3", 
    online_model=False, 
    use_deepinfra=False,
    use_replicate=False,
    repetition_penalty=1.0,
    num_gpus=1, 
    max_new_tokens=512, 
    debug=False,
    device='cuda:0'
)

llm_gen.generation_results()


import ipdb
ipdb.set_trace()

A = [ 1,2,3]