import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from arguments import parse_option


if __name__ == "__main__":
    args = parse_option()

    gene_name = np.load(f'{args.output_dir}/{args.fasta_file}/gene_name.npy')
    gene_data = np.load(f'{args.output_dir}/{args.fasta_file}/gene_data_bin{args.bin}_walk{args.walk}_origin.npy')
    
    TF_dict = open(args.data_dir+'TF_dict.txt', 'r').read().split('\n')
    TF_name_list = [TF_name for TF_name in TF_dict if TF_name]

    pred_list = []
    for _, TF_name in tqdm(enumerate(TF_dict), leave=False):
        if TF_name:
            pred = np.load(f'{args.output_dir}/{args.fasta_file}/1st-pred/{TF_name}.npy')
            pred = (pred*100).round().astype(np.int8)
            pred_list.append(pred)
    pred = np.array(pred_list)
    np.save(f'{args.output_dir}/{args.fasta_file}/1st-pred/pred_all.npy', np.array(pred))
    
    p = Path(f'{args.output_dir}/{args.fasta_file}/1st-pred-csv/')
    p.mkdir(parents=True, exist_ok=True)

    pred = np.load(f'{args.output_dir}/{args.fasta_file}/1st-pred/pred_all.npy')
    pred = pred.transpose((1, 2, 0))
    for i in tqdm(range(len(gene_name)), leave=False):
        data = np.round(pred[i]*0.01, decimals=2)
        columns = TF_name_list
        index = gene_data[i]
        df = pd.DataFrame(data=data, columns=columns, index=index)
        df.to_csv(f'{args.output_dir}/{args.fasta_file}/1st-pred-csv/{gene_name[i]}.csv')


