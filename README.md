# PyPhi: Code Completion for Python

This is a demo of how to train an LLM and use it to complete your Python code snippets.

The model is based on Microsoft's recent [Phi-2](https://huggingface.co/microsoft/phi-2) model. Thus, it's called PyPhi. It's trained on the [ArtifactAI/arxiv_python_research_code](https://huggingface.co/datasets/ArtifactAI/arxiv_python_research_code) dataset.


You can find the training progress [here](https://api.wandb.ai/links/chenghuzi/c1a9vy8m).

The model checkpoint is served on a huggingface repo [chenghuzi/pyphi](https://huggingface.co/chenghuzi/pyphi).

## Quickstart

Clone the repo first. Make sure you have at least python 3.10 and cuda installed.

Install dependencies:

```
pip install -r requirements.txt
```

### Run the model with weights from from 🤗

```bash
$ python predict.py inference --chkpt chenghuzi/pyphi-p
loading base model from microsoft/phi-2
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.70s/it]
loading adapter from chkpts
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
> print('hell
completion:
print('helloworld'[::-1])

>
```

Btw, you can run `/exit` to exit the program.


### Train and test locally

#### Configuration

Creat a config file `config.yml` in the project dir:

```yaml
finetune:
  hub_model_id: chenghuzi/pyphi-p # huggingface model hub id
  push: true # set to false if you don't want to upload
  dev: true # set to false to use mode data
  max_length: 80 # be care about this if you have limited resource
  batch_size: 4 # same as above
  learning_rate: 0.00002
  train_epochs: 3
  save_steps: 1000
  optim: paged_adamw_8bit
  cores: 4 # parallel processing num
inference:
  max_length: 20 # completion length
  chkpt: chkpts

```



Before training, first login to hugigngface with your cli, if you set `push` to true:

```bash
huggingface-cli login
```

Then use your `config.yml`'s `finetune` to start a training process:

```bash
python train.py finetune
```

After training, to make inference with local checkpoint, just run:

```bash
python predict.py inference --chkpt /path/to/your/checkpoint
```

### Quantization

4-bit quantization is used.

### Next steps

* Replace the manual prompt with prefix tokens
* Introduce context information like file tree and object hierarchy, etc.
  * Need read permission to the project dir
  * Need to read information from LSP etc.