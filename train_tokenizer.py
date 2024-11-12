from utils import check_folder_and_solve
import os
import argparse
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import sentencepiece as spm
import json
import yaml
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./tokenizers",
        help="Path to the output directory. Default is './tokenizers'. ",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="YAML file with the configuration for the training. ",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the main output folder. ",
    )
    parser.add_argument(
        "--training_framework",
        type=str,
        default="sentencepiece",
        help="Framework with which to train the tokenizer. Options are: sentencepiece. SOON: HuggingFace. ",  # TODO: add HuggingFace
    )
    parser.add_argument(
        "--force_overwrite",
        default=False,
        action='store_true',
        help="Force overwrite if output folder already exists. Recommended if run on background. "
             "Default is False. ",
    )
    return parser.parse_args()


def main():
    # Load arguments and check
    args = parse_args()
    if args.training_framework == "huggingface":
        raise NotImplementedError(f"Framework huggingface deprecated. NEEDS TO BE UPDATED. ")
    
    # Load configuration and check
    REQUIRED_ARGUMENTS = ("vocab_size", "model_type", "input")
    with open(args.config_file, "r") as i_f:
        config = yaml.safe_load(i_f)
        for argument in REQUIRED_ARGUMENTS:
            if argument not in config:
                raise KeyError(f"Missing {argument} in config file. ")
    
    # Prepare new tokenizer folder
    os.makedirs(args.output_directory, exist_ok=True)  # Creates full path to directory if it does not exist
    new_tokenizer_folder = os.path.join(args.output_directory, args.name)
    check_folder_and_solve(new_tokenizer_folder, force=args.force_overwrite)

    # Prepare user_defined_symbols and control_symbols
    user_defined_symbols = config["user_defined_symbols"]
    control_symbols = config["control_symbols"]
    whitespaces_tokens = ["▁" * i for i in range(2, config["whitespaces_sequences"] + 1)]
    tabulations_tokens = ["\t" * i for i in range(2, config["tabulations_sequences"] + 1)]
    newlines_tokens = ["\n" * i for i in range(2, config["newlines_sequences"] + 1)]
    reserved_tokens = [config["reserved_tokens_format"].replace("{{REPLACE}}", str(i)) for i in range(1, config["n_reserved_tokens"] + 1)]
    user_defined_symbols = user_defined_symbols + whitespaces_tokens + tabulations_tokens + newlines_tokens
    if config["reserved_tokens_as_user_defined_symbols"]:
        user_defined_symbols = user_defined_symbols + reserved_tokens
    else:
        control_symbols = control_symbols + reserved_tokens

    # Prepare training arguments
    model_prefix = os.path.join(new_tokenizer_folder, args.name)
    config["user_defined_symbols"] = user_defined_symbols
    config["control_symbols"] = control_symbols
    config["model_prefix"] = model_prefix
    UNWANTED_ARGUMENTS = ("whitespaces_sequences", "tabulations_sequences", "newlines_sequences", "n_reserved_tokens", "reserved_tokens_format", "reserved_tokens_as_user_defined_symbols")
    for argument in UNWANTED_ARGUMENTS:
        del config[argument]

    spm.SentencePieceTrainer.train(**config)
    
    print("Saving new tokenizer data...")
    shutil.copy(args.config_file, os.path.join(new_tokenizer_folder, "config.yml"))
    tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(new_tokenizer_folder, f"{args.name}.model"))
    vocabulary = {id_: tokenizer.id_to_piece(id_) for id_ in range(tokenizer.vocab_size())}
    with open(os.path.join(new_tokenizer_folder, "vocab.json"), "w") as v_f:  # save metadata of tokenizer
        json.dump(vocabulary, v_f, indent=4)


if __name__ == "__main__":
    main()
