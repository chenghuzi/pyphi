"""
Let's use MS's phi-2b architecture https://huggingface.co/microsoft/phi-2
"""
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# compute_dtype = torch.float16
compute_dtype = torch.float16
base_model_id = "microsoft/phi-2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# the prompt used to improve performance
prompt = 'Complete the following code snippet, which may have hidden parts before its beginning:\n'


def creat_model_tokenizer4training(dev, lora_r=32) -> Tuple[AutoModelForCausalLM, AutoTokenizer, list]:

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},)

    lora_f = not dev

    if lora_f:
        target_modules = ['v_proj', 'k_proj',
                          'q_proj', 'dense', 'fc1', 'fc2']
        print('fully lora')
    else:
        target_modules = ['layers.0.self_attn.v_proj',
                          'layers.0.self_attn.k_proj',
                          'layers.0.self_attn.q_proj',
                          ]
        print('minimal lora')

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.01,
        bias="none",
        modules_to_save=["lm_head", "embed_tokens"],
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=False)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = tokenizer.encode(prompt)

    return model, tokenizer, prompt_tokens


def creat_model_tokenizer4inference(adapter_model) -> Tuple[AutoModelForCausalLM, AutoTokenizer, list]:
    print(f'loading base model from {base_model_id}')
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},)
    print(f'loading adapter from {adapter_model}')
    model = PeftModel.from_pretrained(model, adapter_model)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = tokenizer.encode(prompt)

    return model, tokenizer, prompt_tokens


if __name__ == '__main__':
    creat_model_tokenizer4training(True)
