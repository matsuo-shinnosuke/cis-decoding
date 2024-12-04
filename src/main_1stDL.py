import pickle
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from model import get_FC_3layer
from arguments import parse_option
from utils import dna2onehot

def load_data(data_dir, fasta_file_name, gene_length, output_dir):
    if os.path.exists(f'{output_dir}/{fasta_file_name}/gene_name.npy'):
        print('loading data ...')
        name = np.load(f'{output_dir}/{fasta_file_name}/gene_name.npy')
        data = np.load(f'{output_dir}/{fasta_file_name}/gene_data.npy')
        data_origin = np.load(f'{output_dir}/{fasta_file_name}/gene_data_origin.npy')
        print('[gene_name.npy] size: ' + str(name.shape))
        print('[gene_data.npy] size: ' + str(data.shape))
        print('[gene_data_origin.npy] size: ' + str(data_origin.shape))
    else:
        print('.fa->.npy file ...')
        fas = open(f'{data_dir}/{fasta_file_name}', 'r')
        fas = fas.readlines()

        name, data_origin, data = [], [], []
        for i in tqdm(range(0, len(fas), 2), leave=False):
            if len(fas[i+1].rstrip()) == gene_length:
                name.append(fas[i].rstrip()[1:])
                data_origin.append(fas[i+1].rstrip())
                data.append(dna2onehot(
                    list(fas[i+1].rstrip())).astype(np.int8))
            else:
                print(fas[i].rstrip(), len(fas[i+1].rstrip()))
                print(f'Not satisfy gene length {gene_length}')
        

        p = Path(f'{output_dir}/{fasta_file_name}/')
        p.mkdir(parents=True, exist_ok=True)

        name = np.array(name)
        np.save(f'{output_dir}/{fasta_file_name}/gene_name.npy', name)
        print('[gene_name.npy] size: ' + str(name.shape))
        data = np.array(data)
        np.save(f'{output_dir}/{fasta_file_name}/gene_data.npy', data)
        print('[gene_data.npy] size: ' + str(data.shape))
        data_origin = np.array(data_origin)
        np.save(f'{output_dir}/{fasta_file_name}/gene_data_origin.npy', data_origin)
        print('[gene_data_origin.npy] size: ' + str(data_origin.shape))

    return name, data, data_origin


def split_bin(X, X_origin, output_dir, fasta_file_name, bin, walk):
    X_bin_path = f'{output_dir}/{fasta_file_name}/gene_data_bin{bin}_walk{walk}.npy'
    X_bin_origin_path = f'{output_dir}/{fasta_file_name}/gene_data_bin{bin}_walk{walk}_origin.npy'

    if os.path.exists(X_bin_path):
        print('load %s ...' % (X_bin_path))
        X_bin = np.load(X_bin_path)
        print(f'[gene_data_bin{bin}_walk{walk}.npy] size: ' + str(X_bin.shape))
        X_bin_origin = np.load(X_bin_origin_path)
        print(f'[gene_data_bin{bin}_walk{walk}_origin.npy] size: ' + str(X_bin_origin.shape))

    else:
        print('split ...')
        length = len(X[0])
        X_bin, X_bin_origin = [], []
        for i in tqdm(range(len(X)), leave=False):
            n = 0
            sub, sub_origin = [], []
            while length >= bin + (n*walk):
                sub.append(X[i][n*walk: bin + n*walk])
                sub_origin.append(X_origin[i][n*walk: bin + n*walk])
                n += 1

            X_bin.append(np.array(sub))
            X_bin_origin.append(np.array(sub_origin))
        X_bin, X_bin_origin = np.array(X_bin), np.array(X_bin_origin)
        np.save(X_bin_path, X_bin)
        print(f'[gene_data_bin{bin}_walk{walk}.npy] size: ' + str(X_bin.shape))
        np.save(X_bin_origin_path, X_bin_origin)
        print(f'[gene_data_bin{bin}_walk{walk}_origin.npy] size: ' + str(X_bin_origin.shape))
    return X_bin, X_bin_origin

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        return data

def prediction(model, dataloader, device):
    model.eval()
    pred = []
    with torch.no_grad():
        for x in tqdm(dataloader, leave=False):
            x = x.to(device)
            y = model(x)
            prob = F.softmax(y, dim=-1)[:, :, 1]
            pred.extend(prob.cpu().detach().numpy())
    return np.array(pred)


def detect_peak(pred, bin, threshold):
    # padding
    len_gene = pred.shape[1]
    n_pad = bin - (len_gene % bin)
    pred = np.pad(pred, [(0, 0), (0, n_pad)], 'constant')
    # split
    pred = pred.reshape(pred.shape[0], -1, bin)
    # thereshold process
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    pred = pred.sum(axis=2)
    pred[pred != 0] = 1
    return pred.astype(np.int8)


if __name__ == "__main__":
    args = parse_option()

    ######## Loading data #########
    name, data, data_origin = load_data(data_dir=args.data_dir, 
                           fasta_file_name=args.fasta_file, 
                           gene_length=args.length,
                           output_dir=args.output_dir)
    TF_dict = open(args.data_dir+'TF_dict.txt', 'r').read().split('\n')

    ######## Splitting bin #########
    gene_data_bin, gene_data_origin_bin = split_bin(X=data, 
                                                    X_origin=data_origin,
                                                    output_dir=args.output_dir,
                                                    fasta_file_name=args.fasta_file,
                                                    bin=args.bin, 
                                                    walk=args.walk)
    
    ######## Prediction #########
    
    p = Path(f'{args.output_dir}/{args.fasta_file}/1st-pred/')
    p.mkdir(parents=True, exist_ok=True)
    p = Path(f'{args.output_dir}/{args.fasta_file}/1st-cis-{args.bin_peak}-{args.threshold}/')
    p.mkdir(parents=True, exist_ok=True)

    model = get_FC_3layer(bin=args.bin).to(args.device)
    for ID, TF_name in enumerate(TF_dict):
        print(ID, TF_name)

        ################################################################
        p = f'{args.output_dir}/{args.fasta_file}/1st-pred/{TF_name}.npy'

        model.load_state_dict(torch.load(
            f'{args.model_dir}/{TF_name}.pkl',
            map_location=torch.device(args.device)))
        
        dataset = Dataset(gene_data_bin)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers)
        pred = prediction(model, dataloader, args.device)
        np.save(p, pred)

        ################################################################
        p = Path(f'{args.output_dir}/{args.fasta_file}/1st-cis-{args.bin_peak}-{args.threshold}/{TF_name}.npy')
        pred = np.load(f'{args.output_dir}/{args.fasta_file}/1st-pred/{TF_name}.npy')
        peak = detect_peak(pred, args.bin_peak, args.threshold)
        np.save(p, peak)