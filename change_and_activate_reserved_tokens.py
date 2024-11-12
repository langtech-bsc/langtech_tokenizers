from transformers.convert_slow_tokenizer import import_protobuf
import transformers
import re
import os
import yaml
import argparse
from utils import check_folder_and_solve
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Main directory to save new tokenizer. ",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="YAML file with the information of the source model and the changes. ",
    )
    parser.add_argument(
        "--tokenizer_class",
        type=str,
        default="LlamaTokenizer",
        help="HuggingFace tokenizer class with which the Sentencepiece model will be loaded. Default is LlamaTokenizer. ",
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
    args = parse_args()
    
    # import specified tokenizer class
    TokenizerClass = getattr(transformers, args.tokenizer_class)

    with open(args.config_file, "r") as i_f:
        config = yaml.safe_load(i_f)

    model_pb2 = import_protobuf()
    source_model_proto = model_pb2.ModelProto()
    with open(config["source_model"], "rb") as f:
        source_model_proto.ParseFromString(f.read())
         
    # get vocab 
    tokenizer_vocab = set()
    for piece in source_model_proto.pieces:
        tokenizer_vocab.add(str(piece.piece))
        if piece.piece in config["tokens_to_add"]:  # if already in tokenizer, just change type
            piece.type = config["tokens_to_add"][piece.piece]
    
    pattern = re.compile(config["reserved_token_pattern"])
    for i, (token_to_add, type_) in enumerate(config["tokens_to_add"].items()):
        # if already in tokenizer, only change type
        if token_to_add in tokenizer_vocab:
            continue
        found = False
        for piece in source_model_proto.pieces:
            if pattern.match(str(piece.piece)):
                piece.piece = token_to_add
                piece.type = type_
                found = True
                break
        if not found:
            raise Error(f"Not enough reserved tokens found matching the pattern ({i} found but {len(config['tokens_to_add'])} are needed). Aborting...")
    
    os.makedirs(os.path.dirname(args.output_directory), exist_ok=True)
    check_folder_and_solve(args.output_directory, force=args.force_overwrite)
    path_to_file = os.path.join(args.output_directory, "tokenizer.model")
    with open(path_to_file, 'wb') as o_f:
        o_f.write(source_model_proto.SerializeToString())  # This will be replaced afterwards by the last tokenizer.json
    
    new = TokenizerClass.from_pretrained(path_to_file)
    new.add_special_tokens({"additional_special_tokens": list(config["tokens_to_add"].keys())})
    new.save_pretrained(args.output_directory)
    # copy original config
    if config["original_config"]:
        shutil.copyfile(config["original_config"], os.path.join(args.output_directory, os.path.basename(config["original_config"])))
    # copy changes config
    shutil.copyfile(args.config_file, os.path.join(args.output_directory, "changes.yml"))


if __name__ == "__main__":
    main()
