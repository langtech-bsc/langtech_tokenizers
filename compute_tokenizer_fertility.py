from transformers import GPT2TokenizerFast
import os
from tqdm import tqdm
from datasets import load_dataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sentencepiece as spm 
import sys
import regex

 
def add_base_arguments_to_parser(parser):
 
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to dataloader. '
    )
    parser.add_argument(
        '--dataset_split',
        type=str,
        default="train",
        help='Dataset split name. default is "train". '
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to tokenizer.'
    )
    parser.add_argument(
        '--tokenizer_framework',
        type=str,
        default="sentencepiece",
        help='Framework with which the tokenizer was trained. Options are: huggingface and sentencepiece. Default is sentencepiece. '
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output file. '
    )


def main():
    
    description = """ Compute Tokens per Word """
    parser = ArgumentParser(description=description, formatter_class=ArgumentDefaultsHelpFormatter)
    add_base_arguments_to_parser(parser)
    args = parser.parse_args()

    def n_words(sentence):
        # words = sentence.split()  DEPRECATED
        words = regex.findall(r"[ ]?[\p{L}]+|[ ]?[^\p{L}\p{N} \t\n]+|[ ]+|[\t]+|[\n]+|\d{1}", sentence) 
        return len(words)
    
    def n_bytes(sentence):
        bytes_ = len(sentence.encode('utf-8'))
        return bytes_

    print("Preparing dataset ...")
    dataset = load_dataset(args.dataset_path, split=args.dataset_split)

    print(f"DATASET: {dataset}")
    total_words = 0
    total_bytes = 0
    for example in dataset:
        total_words += n_words(example["text"])
        total_bytes += n_bytes(example["text"])

    print("Loading tokenizer ...")
    if args.tokenizer_framework == "huggingface":
        tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
        
        def tokenization(example_):
            return tokenizer(example_["text"])
    
    elif args.tokenizer_framework == "sentencepiece":
        tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        
        def tokenization(example_):
            return {"input_ids": tokenizer.encode(example_["text"])}
    
    else:
        raise NotImplementedError(f"Framework {args.tokenizer_framework} not recognized. ")

    print("Tokenizing ...")
    new_dataset = dataset.map(tokenization, batched=True)
    total_tokens = 0
    for example in tqdm(new_dataset):
        example_tokens = len(example["input_ids"])
        total_tokens += example_tokens
    
    print("Writing results")
    tpw = total_tokens / total_words
    tpb = total_tokens / total_bytes
    with open(args.output_file, "w") as o_f:
        o_f.write(f"total_words,total_tokens,total_bytes,tpw,tpb\n")
        o_f.write(f"{total_words},{total_tokens},{total_bytes},{tpw},{tpb}\n")
    print("DONE!")


if __name__ == "__main__":
    main()

