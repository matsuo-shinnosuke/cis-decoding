import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from glob import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import pickle
from pathlib import Path

from arguments import parse_option
from model import TransformerClassification
from utils import *

if __name__ == "__main__":
    args = parse_option()

    X = np.load(f'{args.output_dir}/{args.fasta_file}/2nd-data.npy')
    Y = np.load(f'{args.output_dir}/{args.fasta_file}/2nd-label.npy')
    gene_name = np.load(f'{args.output_dir}/{args.fasta_file}/gene_name.npy')
    # # only test data
    # _, X, _, Y, _, gene_name = train_test_split(
    #     X, Y, gene_name, test_size=0.33, random_state=args.seed)
    X, gene_name = X[Y==1], gene_name[Y==1] # expression data
    
    ##### Inference ######
    model = TransformerClassification(model_dim=X.shape[2], max_seq_len=X.shape[1]).to(args.device)
    model.load_state_dict(torch.load(f'{args.output_dir}/{args.fasta_file}/model.pkl', map_location=args.device))
    model.eval()
    
    result_list = []
    for idx in tqdm(range(len(X)), leave=False):
        data = torch.tensor(X[idx]).float().to(args.device)
        data = data.unsqueeze(0).requires_grad_()

        _, attention = model(data, return_attention=True)
        attention = attention.mean(1)[:, 0, 1:]
        result_list.append(attention[0].cpu().detach().numpy())

    result = np.array(result_list)
    np.save(f'{args.output_dir}/attention_2ndDL_result', result)

    ###### save ######
    print('Saving results ...')
    df = pd.DataFrame(data=np.round(result, decimals=6), index=gene_name)
    df.to_csv(f'{args.output_dir}/{args.fasta_file}/2nd-attention-weight.csv')