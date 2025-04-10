import os
import argparse

"""
python generate_fertility_report.py --main_folder fertility_results/2024-03-14/
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_folder",
        type=str,
        required=True,
        help="Main folder where the results are saved. "
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    tokenizers = [dir_ for dir_ in os.listdir(args.main_folder) if os.path.isdir(os.path.join(args.main_folder, dir_))]
    datasets = []
    for tokenizer in tokenizers:  # union of all datasets for all tokenizers
        tokenizer_results_path = os.path.join(args.main_folder, tokenizer)
        tokenizer_datasets = [dataset.split(".")[0] for dataset in os.listdir(tokenizer_results_path)]
        for dataset in tokenizer_datasets:
            if dataset not in datasets:
                datasets.append(dataset)
    print(f"Tokenizers included: {tokenizers}")
    print(f"Datasets included: {datasets}")
    
    with open(os.path.join(args.main_folder, f"report_tpw.tsv"), "w") as tpw_f, open(os.path.join(args.main_folder, f"report_tpb.tsv"), "w") as tpb_f:
        tpw_f.write(f"tokenizer")
        tpb_f.write(f"tokenizer")
        for dataset in datasets:
            tpw_f.write(f"\t{dataset}")
            tpb_f.write(f"\t{dataset}")
        tpw_f.write("\n")
        tpb_f.write("\n")
        for tokenizer in tokenizers:
            tpw_f.write(f"{tokenizer}")
            tpb_f.write(f"{tokenizer}")
            tokenizer_folder = os.path.join(args.main_folder, tokenizer)
            for dataset in datasets:
                try: 
                    with open(os.path.join(tokenizer_folder, f"{dataset}.csv"), "r") as i_f:
                        lines = i_f.readlines()
                        tpw = lines[1].split(",")[3].strip()
                        tpb = lines[1].split(",")[4].strip()
                except:
                    tpw = "-"
                    tpb = "-"
                tpw_f.write(f"\t{tpw}")
                tpb_f.write(f"\t{tpb}")
            tpw_f.write("\n")
            tpb_f.write("\n")
    print("Done!")


if __name__ == "__main__":
    main()
