from transformers import LlamaTokenizer, LlamaTokenizerFast
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

 
def add_base_arguments_to_parser(parser):
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to tokenizer "*.model" file.'
    )


def main():
    description = """ Convert SentencePiece tokenizer to HuggingFace """
    parser = ArgumentParser(description=description, formatter_class=ArgumentDefaultsHelpFormatter)
    add_base_arguments_to_parser(parser)
    args = parser.parse_args()

    output_folder = os.path.dirname(args.tokenizer_path)
    hf_folder = os.path.join(output_folder, "hf")

    os.makedirs(hf_folder)
    slow_tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    slow_tokenizer.save_pretrained(hf_folder)  # NOTICE: the only difference of fast from slow is that it adds a tokenizer.json file. You can load slow_tokenizer from fast
    fast_tokenizer = LlamaTokenizerFast.from_pretrained(hf_folder)
    fast_tokenizer.save_pretrained(hf_folder)

    print("DONE!")


if __name__ == "__main__":
    main()

