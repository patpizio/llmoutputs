import torch
from transformers import GenerationConfig

def prompt_llm(prompt, tokenizer, model, temperature=1.0, top_k=100, top_p=0.99, max_new_tokens=1, device='cuda'):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    # input_ids_gpu = input_ids.to(device)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.5,
        max_new_tokens=max_new_tokens,
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
    return inputs, outputs