import copy
import random

import click
import numpy as np
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

import wandb
from model import creat_model_tokenizer4training
from utils import cli


@cli.command()
@click.option('--max_length', type=int)
@click.option('--batch_size', type=int)
@click.option('--learning_rate', type=float)
@click.option('--train_epochs', type=int)
@click.option('--save_steps', type=int)
@click.option('--optim', type=str)
@click.option('--hub_model_id', type=str)
@click.option('--push', type=bool)
@click.option('--cores', type=int)
@click.option('--lora_r', type=int)
@click.option('--dev', type=bool)
@click.pass_context
def finetune(ctx,
             max_length,
             batch_size,
             learning_rate,
             train_epochs,
             save_steps,
             optim,
             hub_model_id,
             push,
             cores,
             lora_r,
             dev
             ):
    chkpts_dir = 'chkpts'

    if dev is True:
        split = 'train[:2000]'
    else:
        split = 'train[:20000]'  # chang this to a larger number in the future
    ds = load_dataset("ArtifactAI/arxiv_python_research_code",
                      split=split).filter(lambda x: len(x['code']) >= 500)
    train_test_split = ds.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    # filter with minimal length 100
    model, tokenizer, prompt_tokens = creat_model_tokenizer4training(
        dev, lora_r=lora_r)

    prompt_tokens_length = len(prompt_tokens)

    def substring_or_original(s):
        if len(s) < 600:
            return s
        else:
            # return s[random.randint(0, len(s)-500):random.randint(500, len(s))]
            return s[random.randint(0, len(s)-500):]

    def tokenize_function(examples):
        all_tokens_batch = []
        attention_mask_batch = []
        for code in examples["code"]:
            # here insert left, middle right
            snippet = substring_or_original(code)
            code_tokens = tokenizer.encode(snippet)
            # here we may also need sampling
            all_tokens = (prompt_tokens + code_tokens)[:max_length]
            completion_point = np.random.randint(
                prompt_tokens_length+1, len(all_tokens))
            # tokenizer.decode(all_tokens[:completion_point])
            # tokenizer.decode(all_tokens[completion_point:])
            attention_mask = [0 for _ in range(
                completion_point)] + [1 for _ in range(len(all_tokens) - completion_point)]
            length_diff = max_length - len(all_tokens)

            if length_diff > 0:
                all_tokens = [tokenizer.pad_token_id for _ in range(
                    length_diff)] + all_tokens
                attention_mask = [0 for _ in range(
                    length_diff)] + attention_mask
            all_tokens_batch.append(copy.deepcopy(all_tokens))
            attention_mask_batch.append(copy.deepcopy(attention_mask))

        all_tokens_batch = torch.Tensor(
            all_tokens_batch).type(torch.LongTensor)
        attention_mask_batch = torch.Tensor(
            attention_mask_batch).type(torch.LongTensor)
        labels = torch.cat([all_tokens_batch[:, 1:], torch.zeros(
            all_tokens_batch.shape[0], 1).type(torch.LongTensor)-100], dim=-1)

        return {
            'input_ids': all_tokens_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels,
        }
    mbs = 16
    tokenized_train_dataset = train_dataset.map(
        tokenize_function, batched=True,
        batch_size=mbs,
        num_proc=cores,
        load_from_cache_file=not dev)
    tokenized_test_dataset = test_dataset.map(
        tokenize_function, batched=True,
        batch_size=mbs,
        num_proc=cores,
        load_from_cache_file=not dev)

    if dev is False:
        wandb.init(project="pyphi", config={
            "max_length": max_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "train_epochs": train_epochs,
            "model_name": "pyphi"
        })

    training_args = TrainingArguments(
        output_dir=chkpts_dir,
        evaluation_strategy="epoch",
        overwrite_output_dir=True,
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=6,
        optim=optim,
        save_steps=save_steps,
        report_to='none' if dev else 'wandb',
        logging_strategy="steps",
        logging_steps=save_steps//10,
        hub_model_id=hub_model_id,
        max_grad_norm=10.0,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
    )

    model.config.use_cache = False
    trainer.train()
    if push:
        trainer.push_to_hub()


if __name__ == '__main__':
    cli()
