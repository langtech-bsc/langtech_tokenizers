from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from fastai.text.all import Transform, TitledStr, tensor
from torch import nn
import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import argparse
import sys
import os
import numpy as np  # no problem
from utils import check_folder_and_solve
import scipy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--small_source_model_directory",
        type=str,
        help="Main folder of the small source model."
    )
    parser.add_argument(
        "--big_source_model_directory",
        type=str,
        required=True,
        help="Main folder of the big source model."
    )
    parser.add_argument(
        "--small_target_model_directory",
        type=str,
        help="Main folder of the small target model."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Whether the model is an 'encoder' or 'decoder'."
    )
    parser.add_argument(
        "--untied_embeddings",
        default="False",
        action="store_true",
        help="Whether the embeddings and unembeddings of the model are tied."
    )
    parser.add_argument(
        "--source_tokenizer",
        type=str,
        default=None,
        help="Main folder of the source tokenizer. Default is the same folder as big source model. "
    )
    parser.add_argument(
        "--target_tokenizer",
        type=str,
        default=None,
        help="Main folder of the target tokenizer. Default is the same folder as small target model (if given)."
    )
    parser.add_argument(
        "-o", "--output_directory",
        type=str,
        default="./new_models",
        help="Path to the output directory. "
             "Default is './new_models'.",
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="Name of the new model. "
             "Default is 'new_{base_model_name}'.",
    )
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default="matching",
        help="Which strategy we want to use. Options are:\n"
             " - matching (default)\n"
             " - improved\n"
             " - lstsq\n"
             " - orthogonal_procrustes\n"
             " - knn (NOT IMPLEMENTED)\n"
             " - wechsel (NOT IMPLEMENTED)",
    )
    parser.add_argument(
        "--pad_token",
        type=str,
        default=None,
        help="Optionally, you can set the pad token in the new tokenizer.",
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        default=None,
        help="Optionally, you can set the pad token in the new tokenizer.",
    )
    parser.add_argument(
        "--bos_token",
        type=str,
        default=None,
        help="Optionally, you can set the pad token in the new tokenizer.",
    )
    parser.add_argument(
        "-d", "--debug",
        default=False,
        action='store_true',
        help="Faster, full models are not loaded, only embeddings (if possible). "
             "Results are not saved, but a console is opened at the end of run for further testing. "
             "NOT RECOMMENDED for anything else then debugging. "
             "Default is False.",
    )
    parser.add_argument(
        "-f", "--force_overwrite",
        default=False,
        action='store_true',
        help="Force overwrite if output folder already exists. Recommended if run on background. "
             "Default is False.",
    ) 
    parser.add_argument(
        "--save_embeddings",
        default=False,
        action='store_true',
        help="Save embeddings of all models for future use.",
    )
    return parser.parse_args()


class Warning_message():

    def __init__(self, msg):
        print(f"*WARNING*: {msg}")


class TransformersTokenizer(Transform):

    def __init__(self, tokenizer):
        super(TransformersTokenizer, self).__init__()
        self.tokenizer = tokenizer

    def encodes(self, x):
        tokens = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(tokens))

    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


def load_model_embeddings(model_path, model_type, return_unembeddings=False, return_model=False):
    model_weights_input = None
    model_weights_output = None
    
    # if they were already saved
    embedding_path = os.path.join(model_path, "embeddings.pt")
    unembedding_path = os.path.join(model_path, "unembeddings.pt")
    if not return_model and os.path.exists(embedding_path):
        model_weights_input = torch.load(embedding_path)
        if return_unembeddings and os.path.exists(unembedding_path):
            model_weights_output = torch.load(unembedding_path)

    # not pre-saved or full model is required
    need2load = (model_weights_input is None) or (model_weights_output is None and return_unembeddings) or return_model
    if need2load:
        if model_type == 'encoder':
            model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)  # here we save the big source model and we will apply to the same modifications to make it the big target model
        elif model_type == 'decoder':
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        else:
            raise ValueError(f"Invalid model type {model_type}. Options: 'encoder' or 'decoder'.")
        
        model_weights_input = model.get_input_embeddings().weight.clone().detach()
        if return_unembeddings:
            model_weights_output = model.get_output_embeddings().weight.clone().detach()

        if model_type == 'encoder':  # in encoders, unembeddings have bias
            bias = model.get_output_embeddings().bias
            model_weights_input = torch.cat((model_weights_input, bias[:, None]),1)  # attach at the end
    
    # return
    if return_model:
        return model_weights_input, model_weights_output, model
    return model_weights_input, model_weights_output


def ort_proc_transformation(small_source_embeddings, big_source_embeddings, small_target_embeddings):
    padding = ((0, 0), (0, big_source_embeddings.shape[1] - small_source_embeddings.shape[1]))
    small_source_tmp = np.pad(small_source_embeddings, padding)
    small_target_tmp = np.pad(small_target_embeddings, padding)
    w, sca = scipy.linalg.orthogonal_procrustes(small_source_tmp, big_source_embeddings)
    big_target_embeddings = small_target_tmp @ w
    return torch.tensor(big_target_embeddings), sca


def lstsq_transformation(small_source_embeddings, big_source_embeddings, small_target_embeddings):
    w, residuals, _, _ = np.linalg.lstsq(small_source_embeddings, big_source_embeddings, rcond=None)
    big_target_embeddings = np.dot(small_target_embeddings, w)
    return torch.tensor(big_target_embeddings), residuals


def get_tensor_index(tensor_, value) -> int:
    int((tensor_ == value).nonzero(as_tuple=False).squeeze())


def save_embeddings(embeddings_tensor, model_path, unembedding_tensor=None, force=False):
    print("\tSaving embeddings ...")
    destination_path = os.path.join(model_path, "embeddings.pt")
    if not force and os.path.exists(destination_path):
        print(f"\tEmbeddings of model {model_path} are already saved. Skipping. "
              f"If you want to overwrite them, use the --force_overwrite option.")
    else:
        torch.save(embeddings_tensor, destination_path)
        print("\tEmbeddings saved!")
    if unembedding_tensor is not None:
        print("\tSaving unembeddings ...")
        destination_path = os.path.join(model_path, "unembeddings.pt")
        if not force and os.path.exists(destination_path):
            print(f"\tUnembeddings of model {model_path} are already saved. Skipping. "
                  f"If you want to overwrite them, use the --force_overwrite option.")
        else:
            torch.save(unembeddings_tensor, destination_path)
            print("\tUnembeddings saved!")


def main(args):
    # check output folder ready
    name = args.name if args.name else f"new_{os.path.basename(args.big_source_model_directory)}_{args.strategy}"
    new_model_directory = os.path.join(args.output_directory, name)
    check_folder_and_solve(new_model_directory, force=args.force_overwrite)
    assert not args.untied_embeddings or (args.strategy == "matching" and args.model_type == "decoder"), \
        ("NOT IMPLEMENTED: untied_embeddings currently only works with `matching` strategy and `decode` model type. "
        "If you need it, talk with Seve. ")
    
    # LOAD AND INITIALIZE TOKENIZERS
    # load tokenizers
    print("Preparing tokenizers ...")
    if args.source_tokenizer is None:
        args.source_tokenizer = args.big_source_model_directory
    if args.target_tokenizer is None:
        args.target_tokenizer = args.small_source_model_directory
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)
    
    # update special tokens if required
    if args.pad_token:
        target_tokenizer.pad_token = args.pad_token
    if args.eos_token:
        target_tokenizer.eos_token = args.eos_token
    if args.bos_token:
        target_tokenizer.bos_token = args.bos_token

    # check tokenizers' config match
    if target_tokenizer.pad_token != source_tokenizer.pad_token:
        Warning_message(f"PAD token is different in the two tokenizers:\n - Source: {source_tokenizer.pad_token}\n - Target: {target_tokenizer.pad_token}\nIt will be updated in target")
    if target_tokenizer.eos_token != source_tokenizer.eos_token:
        Warning_message(f"EOS token is different in the two tokenizers:\n - Source: {source_tokenizer.eos_token}\n - Target: {target_tokenizer.eos_token}\nIt will be updated in target")
    if target_tokenizer.bos_token != source_tokenizer.bos_token:
        Warning_message(f"BOS token is different in the two tokenizers:\n - Source: {source_tokenizer.bos_token}\n - Target: {target_tokenizer.bos_token}\nIt will be updated in target")
    if target_tokenizer.model_max_length != source_tokenizer.model_max_length:
        Warning_message(f"Different model max length in the two tokenizers:\n - Source: {source_tokenizer.model_max_length}\n - Target: {target_tokenizer.model_max_length}")
    
    # tokenizers: HF -> fastai
    fastai_source_tokenizer = TransformersTokenizer(source_tokenizer)
    fastai_target_tokenizer = TransformersTokenizer(target_tokenizer)
    
    # LOAD MODELS EMBEDDINGS
    print("Loading big source model ...")
    # here we save the big source model to which we will be applying modifications to make it the big target model
    if args.debug:
        big_source_weights_input, big_source_weights_output = load_model_embeddings(args.big_source_model_directory, args.model_type, return_unembeddings=args.untied_embeddings)
    else:
        big_source_weights_input, big_source_weights_output, big_model = load_model_embeddings(args.big_source_model_directory, args.model_type, return_unembeddings=args.untied_embeddings, return_model=True)
    if args.save_embeddings:
        save_embeddings(big_source_weights_input, args.big_source_model_directory, unembedding_tensor=big_source_weights_output, force=args.force_overwrite)
    print("\tBig source model loaded!")
    if args.strategy == "transformation":
        print("Loading small source model ...")
        small_source_weights_input = load_model_embeddings(args.small_source_model_directory, args.model_type)
        if args.save_embeddings:
            save_embeddings(small_source_weights_input, args.small_source_model_directory, force=args.force_overwrite)
        print("\tSmall source model embeddings loaded!")
    if args.strategy in ("transformation", "improved"):
        print("Loading small target model ...")
        small_target_weights_input = load_model_embeddings(args.small_target_model_directory, args.model_type)
        if args.save_embeddings:
            save_embeddings(small_target_weights_input, args.small_target_model_directory, force=args.force_overwrite)
        print("\tSmall target model embeddings loaded!")

    # EMBEDDING ADAPTATION
    target_vocab_size = fastai_target_tokenizer.tokenizer.vocab_size
    big_source_model_size = big_source_weights_input.size(1)  # h: hidden_size
    if args.strategy == "lstsq":
        print("Least Squares in progress ...")
        new_weights_input, residuals = lstsq_transformation(small_source_weights_input, big_source_weights_input,
                                                            small_target_weights_input)
    elif args.strategy == "orthogonal_procrustes":
        print("Orthogonal Procrustes in progress ...")
        new_weights_input, sca = ort_proc_transformation(small_source_weights_input, big_source_weights_input,
                                                         small_target_weights_input)
    elif args.strategy == "knn":
        raise NotImplementedError("Sorry, I chose to sleep!")
    elif args.strategy == "wechsel":
        raise NotImplementedError("Sorry, I chose to sleep!")
    else:
        weights_mean_input = big_source_weights_input.mean(0)
        new_weights_input = big_source_weights_input.new_zeros(target_vocab_size, big_source_weights_input.size(1))  # size: (N+M) * h
        if args.untied_embeddings:
            weights_mean_output = big_source_weights_output.mean(0)
            new_weights_output = big_source_weights_output.new_zeros(target_vocab_size, big_source_weights_output.size(1))  # size: (N+M) * hput

        source_vocabulary = fastai_source_tokenizer.tokenizer.get_vocab()
        target_vocabulary = fastai_target_tokenizer.tokenizer.get_vocab()
        
        # checking matching (and different) tokens
        new_ids_matching_tokens = []
        new_ids_different_tokens = []
        print("Matching tokens ...")
        for new_token, idx_new in target_vocabulary.items():
            idx_old = source_vocabulary.get(new_token, -1)
            if idx_old >= 0:  # if in old vocabulary
                new_weights_input[idx_new] = big_source_weights_input[idx_old]
                if args.untied_embeddings:
                    new_weights_output[idx_new] = big_source_weights_output[idx_old]
                new_ids_matching_tokens.append(idx_new)
            else:
                if args.strategy == "matching":
                    new_weights_input[idx_new] = weights_mean_input
                    if args.untied_embeddings:
                        new_weights_output[idx_new] = weights_mean_output
                new_ids_different_tokens.append(idx_new)

        print("Computing new embeddings ...")
        if args.strategy == "improved":
            different_tokens_tensor = torch.tensor(new_ids_different_tokens)  # size: N
            matching_tokens_tensor = torch.tensor(new_ids_matching_tokens)  # size: M
            # N + M = target_vocab_size
            print("\tCosine similarity matrix ...")
            different_embeddings = torch.index_select(small_target_weights_input, 0, different_tokens_tensor)  # size: N * h
            matching_embeddings = torch.index_select(small_target_weights_input, 0, matching_tokens_tensor)  # size: M * h
            cosine_similarity_matrix = pairwise_cosine_similarity(different_embeddings, matching_embeddings)  # size: N * M
            print("\tAux matrix ...")
            # This achieves the same matrix with extra zero columns. The zeros are placed at the positions relative
            # to cosine_similarity between two Different Tokens (columns of zeros)
            aux_matrix = torch.zeros((len(different_tokens_tensor), target_vocab_size))  # size: N * (N+M). All zeros
            aux_matrix[:, matching_tokens_tensor] = cosine_similarity_matrix  # size: N * (N+M)
            print("\tSimilarity matrix ...")
            # Here we achieve the same matrix
            similarity_matrix = torch.eye(target_vocab_size)  # size: (N+M) * (N+M). Identity
            similarity_matrix[different_tokens_tensor, :] = aux_matrix  # size: (N+M) * (N+M)
            print("\tNormalizing ...")
            normalization_matrix = similarity_matrix.sum(dim=-1).unsqueeze(-1)  # size: (N+M) * 1

            # Free RAM space. Else we get Killed (Rest In Peperoni)
            del small_target_weights_input
            del cosine_similarity_matrix
            del aux_matrix
            del big_source_weights_input

            # Each line i of the similarity matrix, is a vector of length vocab_size, relative to the similarity of the
            # token i with all other tokens. Thus, we can use this to make a weighted sum over the other tokens (notice
            # that only columns of matching tokens have non-zero values, since they are the only vectors we know).
            # TODO: careful, possible division by 0 ...
            similarity_matrix = (similarity_matrix / normalization_matrix)  # size: (N+M) * (N+M). Divide rows by their sum to normalize
            print("\tApplying similarity matrix ...")
            new_weights_input = similarity_matrix @ new_weights_input  # size: (N+M) * h

    # debug mode
    if args.debug:
        import code
        code.interact(local=dict(globals(), **locals()))
        Warning_message(f"Since debug mode was activated, models were not loaded and program cannot continue.\nIf you want to save results, remove --debug argument.")
        sys.exit(1)
        
    # use new weights
    if args.model_type == "encoder":
        new_weights_input, bias = torch.split(new_weights_input, new_weights_input.size(1) - 1, 1)
        new_weights_input = new_weights_input.clone().detach()
        bias = torch.squeeze(bias)
        bias = bias.clone().detach()
    # update embeddings
    new_wte = nn.Embedding(target_vocab_size, big_source_weights_input.size(1))  # wte: weight token embeddings
    new_wte.weight.data = new_weights_input
    big_model.set_input_embeddings(new_wte)
    # update unembeddings
    if not args.untied_embeddings:
        new_weights_output = new_weights_input.clone()  # in Falcon, output and input embeddings are the same matrix
    if args.model_type == "encoder":
        new_output_embeddings = nn.Linear(in_features=new_weights_output.size(1), out_features=new_weights_output.size(0), bias=True)
        new_output_embeddings.bias.data = bias
    else:
        new_output_embeddings = nn.Linear(in_features=new_weights_output.size(1), out_features=new_weights_output.size(0), bias=False)
    new_output_embeddings.weight.data = new_weights_output
    big_model.set_output_embeddings(new_output_embeddings)
    # update config
    new_eos_token_id = target_tokenizer.vocab[target_tokenizer.eos_token]
    new_bos_token_id = target_tokenizer.vocab[target_tokenizer.bos_token]
    new_pad_token_id = target_tokenizer.vocab[target_tokenizer.pad_token] if target_tokenizer.pad_token else None
    big_model.config.eos_token_id = new_eos_token_id
    big_model.config.bos_token_id = new_bos_token_id
    if new_pad_token_id is not None:
        big_model.config.pad_token_id = new_pad_token_id
    else:
        Warning_message(f"Target tokenizer has no PAD token. Skipping...")
    big_model.config.vocab_size = target_vocab_size

    # SAVE OUTPUT
    # save new model and tokenizer
    big_model.save_pretrained(new_model_directory)
    target_tokenizer.save_pretrained(new_model_directory)
    # save embedding layer for debug
    if args.save_embeddings:
        save_embeddings(new_weights_input, new_model_directory, unembedding_tensor=new_weights_output)
    # show results
    print("Embeddings were adapted to the new tokenizer.")
    print("\nRESULTS")
    print(f" - {len(new_ids_matching_tokens)} tokens of the new tokenizer (aprox. "
          f"{'{:.2f}'.format(len(new_ids_matching_tokens) / len(target_vocabulary) * 100)}% of all the tokens of the new vocabulary) "
          f"matched with tokens from the original one.")
    print(f" - {len(new_ids_different_tokens)} tokens (aprox. "
          f"{'{:.2f}'.format(len(new_ids_different_tokens) / len(target_vocabulary) * 100)}%) did not.\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)

