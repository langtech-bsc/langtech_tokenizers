# Langtech Tokenizers

This is Langtech' s code for training, evaluating and modifying tokenizers. In particular, it has been used to train Salamandra's tokenizer.

Unless otherwise specified, all the following scripts use tokenizers from the library SentencePiece.

## Set up

Install python 3.12.
Set up a virtual environment and install the requirements: `requirements.txt`.

You are ready!

## How to train a tokenizer

You can train a tokenizer using the script `train_tokenizer.py`. 

First of all you need to prepare a training configuration file, detailing all the tokenizer choices. 
Make a copy of  `training_configs/template.yml` and edit the file as explained in it. 
You can also check the given example `training_configs/example.yml`. 
This will result in `<YOUR_TRAINING_CONFIG>`.
Notice that Salamandra tokenizer training config is also provided `training_configs/salamandra.yml`.

Choose a name `<NAME>` for your tokenizer and the destination folder `<OUTPUT_DIRECTORY>`.
Finally, you can run: `python train_tokenizer.py --output_directory <OUTPUT_DIRECTORY> --config_file <YOUR_TRAINING_CONFIG> --name <NAME>`.

## Compute tokenizer fertility
By fertility, we mean the two following metrics:
- `tpw`, i.e. average tokens-per-word.
- `tpb`, i.e. average tokens-per-byte.
Where words are splitted by using the following pattern: `[ ]?[\p{L}]+|[ ]?[^\p{L}\p{N} \t\n]+|[ ]+|[\t]+|[\n]+|\d{1}`.

In order to compute both `tpw` and `tpb` for a given tokenizer (with path `<TOKENIZER>`) in a given dataset (i.e. a HF dataloader with path `<DATASET>`) you can use the script `compute_tokenizer_fertility.py`.
Choose an output file (`.csv`) `<OUTPUT_FILE>` and run: `python --dataset_path <DATASET> --tokenizer_path <TOKENIZER> --output_file <OUTPUT_FILE>`.

Notice: you can choose the dataloader split with `--dataset_split <SPLIT>`. By default, a SentencePiece tokenizer is expected, but this code can run with HF tokenizers with `--tokenizer_framework huggingface`.

## Add and/or activate tokens
In order to add new tokens to an existing tokenizer by replacing reserved tokens, or activate disabled ones, you can use `change_and_activate_reserved_tokens.py`.

First of all, you need to prepare a changes configuration file.
Make a copy of `changes_configs/template.yml` and edit the file as explained in it.
You can also check the given example `changes_configs/example.yml`.
This will result in `<YOUR_CHANGES_CONFIG>`.
Notice that Salamandra-instruct tokenizer changes config is also provided `changes_configs/salamandra_instructed.yml`.

Choose the destination folder `<OUTPUT_DIRECTORY>` and run: `python change_and_activate_reserved_tokens.py --output_directory <OUTPUT_DIRECTORY> --config_file <YOUR_CHANGES_CONFIG>`.

