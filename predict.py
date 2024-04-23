import sys

import click
import torch

from model import creat_model_tokenizer4inference
from utils import cli

DEVICE = 'cuda:0'


@cli.command()
@click.option('--max_length', type=int)
@click.option('--chkpt', type=str)
@click.pass_context
def inference(ctx,
              max_length,
              chkpt,
              ):

    model, tokenizer, prompt_tokens = creat_model_tokenizer4inference(chkpt)
    model.eval()

    while True:
        code = input('> ')
        if code == '/exit':
            print('Exiting...')
            sys.exit(0)

        code_ids = tokenizer.encode(code)
        input_ids = torch.Tensor(
            prompt_tokens+code_ids).type(torch.LongTensor).unsqueeze(0).to(DEVICE)
        if input_ids.shape[0] > 2048:
            print('Too long! try to use a shorter snippet')
            continue

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(
                    input_ids).type(torch.LongTensor),
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_length)

            completion_tokens = outputs[0, len(prompt_tokens)+1:]

            if completion_tokens[-1] == tokenizer.pad_token_id:
                completion_tokens = completion_tokens[:-1]
            code_completion = tokenizer.decode(completion_tokens)

            print(f'completion:\n{code_completion}')


if __name__ == '__main__':
    cli()
