import pickle
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arguments import parse_option
from utils import dna2onehot


def load_data(data_dir, fasta_file_name, gene_length, output_dir):
    if os.path.exists(f'{output_dir}/{fasta_file_name}/gene_name.npy'):
        print('loading data ...')
        name = np.load(f'{output_dir}/{fasta_file_name}/gene_name.npy')
        data = np.load(f'{output_dir}/{fasta_file_name}/gene_data.npy')
        print('[gene_name.npy] size: ' + str(name.shape))
        print('[gene_data.npy] size: ' + str(data.shape))
    else:
        print('.fa->.npy file ...')
        fas = open(f'{data_dir}/{fasta_file_name}', 'r')
        fas = fas.readlines()

        name, data = [], []
        for i in tqdm(range(0, len(fas), 2), leave=False):
            if len(fas[i+1].rstrip()) == gene_length:
                name.append(fas[i].rstrip()[1:])
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

    return name, data


def split_bin(X, output_dir, fasta_file_name, bin, walk):
    X_bin_path = f'{output_dir}/{fasta_file_name}/gene_data_bin{bin}_walk{walk}.npy'
    if os.path.exists(X_bin_path):
        print('load %s ...' % (X_bin_path))
        X_bin = np.load(X_bin_path)
        print('[gene_data_bin{bin}_walk{walk}.npy] size: ' + str(X_bin.shape))
    else:
        print('split ...')
        length = len(X[0])
        n_sub = int(((length - bin) / walk)+1)
        X_bin = np.zeros((X.shape[0], n_sub, bin, 4), dtype=np.int8)
        for i, line in enumerate(tqdm(X, leave=False)):
            n = 0
            sub = []
            while length >= bin + (n*walk):
                sub.append(line[n*walk: bin + n*walk])
                n += 1

            X_bin[i] = np.array(sub)
        np.save(X_bin_path, X_bin)
        print(f'[gene_data_bin{bin}_walk{walk}.npy] size: ' + str(X_bin.shape))
    return X_bin

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        return data

    
def get_FC_3layer():
    return nn.Sequential(
        nn.Flatten(start_dim=-2, end_dim=-1),
        nn.Linear(31*4, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


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
    name, data = load_data(data_dir=args.data_dir, 
                           fasta_file_name=args.fasta_file, 
                           gene_length=args.length,
                           output_dir=args.output_dir)
    with open(args.data_dir+'TF_dict.pkl', 'rb') as tf:
        TF_dict = pickle.load(tf)

    ######## Splitting bin #########
    gene_data_bin = split_bin(X=data, 
                              output_dir=args.output_dir,
                              fasta_file_name=args.fasta_file,
                              bin=args.bin, 
                              walk=args.walk)
    
    ######## Prediction #########
    
    p = Path(f'{args.output_dir}/{args.fasta_file}/1st-pred/')
    p.mkdir(parents=True, exist_ok=True)
    p = Path(f'{args.output_dir}/{args.fasta_file}/1st-cis-{args.bin_peak}-{args.threshold}/')
    p.mkdir(parents=True, exist_ok=True)

    model = get_FC_3layer().to(args.device)
    for ID, TF_name in TF_dict.items():
        print(ID, TF_name)

        ################################################################
        p = f'{args.output_dir}/{args.fasta_file}/1st-pred/{TF_name}.npy'
        if not os.path.exists(p):
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
        if not os.path.exists(p):
            pred = np.load(f'{args.output_dir}/{args.fasta_file}/1st-pred/{TF_name}.npy')
            peak = detect_peak(pred, args.bin_peak, args.threshold)
            np.save(p, peak)