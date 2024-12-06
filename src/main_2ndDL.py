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
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arguments import parse_option
from model import CNN, TransformerClassification, CvT
from utils import fix_seed, dna2onehot
from main_1stDL import load_data as load_raw_dataset

def load_dataset(args):
    ############# data ################
    print('making dataset..')
    result_1st_path = f'{args.output_dir}/{args.fasta_file}/1st-cis-{args.bin_peak}-{args.threshold}/'
    path_list = glob(result_1st_path+'*.npy')
    path_list.sort()

    data = []
    for path in tqdm(path_list, leave=False):
        data.append(np.load(path))

    data = np.array(data)
    data = data.transpose((1, 2, 0))
    np.save(f'{args.output_dir}/{args.fasta_file}/2nd-data.npy', data)
    print('[2nd-data.npy] size: ' + str(data.shape))

    ############ label ################
    gene_name = np.load(f'{args.output_dir}/{args.fasta_file}/gene_name.npy')

    gt = open(f'{args.data_dir}/{args.label_file}', 'r')
    gt = gt.readlines()
    y_gene_name = [y.split()[0] for y in gt]
    y_data = [int(y.split()[1]) for y in gt]
    label_dict = dict(zip(y_gene_name, y_data))

    label = np.array([label_dict[name] for name in gene_name])
    np.save(f'{args.output_dir}/{args.fasta_file}/2nd-label.npy', label)
    print('[2nd-label.npy] size: ' + str(label.shape))

    return data, label, gene_name

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, name):
        self.data = data
        self.label = label
        self.name = name
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.label[idx]).long()
        name = self.name[idx]
        return data, label, name

if __name__ == "__main__":
    args = parse_option()
    fix_seed(seed=args.seed)

    ######## Loading data #########
    if args.model in ['CNN', 'Transformer']:
        X, Y, gene_name = load_dataset(args)
    elif args.model in ['CvT']:
        X, Y, gene_name = load_dataset(args)
        _, X, _ = load_raw_dataset(data_dir=args.data_dir, 
                            fasta_file_name=args.fasta_file, 
                            gene_length=args.length,
                            output_dir=args.output_dir)
    else:
        print('model none')
    

    ######## Preparation #########
    X_train, X_test, Y_train, Y_test, name_train, name_test = train_test_split(
            X, Y, gene_name, test_size=0.3, random_state=args.seed)

    train_dataset = Dataset(X_train, Y_train, name_train)
    test_dataset = Dataset(X_test, Y_test, name_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    if args.model == 'CNN':
        model = CNN(data_length=X.shape[1], n_channel=X.shape[2]).to(args.device)
    elif args.model == 'Transformer':
        model = TransformerClassification(model_dim=X.shape[2], max_seq_len=X.shape[1], n_layers=args.n_layers).to(args.device)
    elif args.model == 'CvT':
        model = CvT(ch=X.shape[2], length=X.shape[1], n_layers=args.n_layers, bin=args.bin).to(args.device)
    else:
        print('model none')

    weight = 1 / np.eye(2)[Y].sum(axis=0)
    weight /= weight.sum()
    weight = torch.tensor(weight).float().to(args.device)
    loss_function = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ############ Training #############
    time_start = time()
    best_auc = 0
    train_auc_list, test_auc_list = [], []
    train_loss_list, test_loss_list = [], []

    for epoch in range(args.num_epochs):
        ####################################
        model.train()

        losses = []
        train_name = []
        train_gt, train_pred = [], []

        for batch in tqdm(train_loader, leave=False):
            data, label, name = batch[0], batch[1], batch[2]
            data, label = data.to(args.device), label.to(args.device)

            y = model(data)
            loss = loss_function(y, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            train_name.extend(name)
            train_gt.extend(label.cpu().detach().numpy())
            train_pred.extend(F.softmax(y, dim=1)[:, 1].cpu().detach().numpy())

        train_loss = np.array(losses).mean()
        train_auc = roc_auc_score(train_gt, train_pred)

        ####################################
        model.eval()

        losses = []
        test_name = []
        test_gt, test_pred = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False):
                data, label, name = batch[0], batch[1], batch[2]
                data, label = data.to(args.device), label.to(args.device)

                y = model(data)
                loss = loss_function(y, label)
                losses.append(loss.item())

                test_name.extend(name)
                test_gt.extend(label.cpu().detach().numpy())
                test_pred.extend(F.softmax(y, dim=1)[:, 1].cpu().detach().numpy())

        test_loss = np.array(losses).mean()
        test_auc = roc_auc_score(test_gt, test_pred)

        ####################################
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_auc_list.append(train_auc)
        test_auc_list.append(test_auc)

        plt.plot(train_auc_list, label='training AUC')
        plt.plot(test_auc_list, label='test AUC')
        plt.title('AUC')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        plt.grid()
        plt.legend()
        plt.savefig(f'{args.output_dir}/{args.fasta_file}/curve_auc.png', dpi=300)
        plt.close()

        plt.plot(train_loss_list, label='training loss')
        plt.plot(test_loss_list, label='test loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.grid()
        plt.legend()
        plt.savefig(f'{args.output_dir}/{args.fasta_file}/curve_loss.png', dpi=300)
        plt.close()
        
        print('[%d/%d]: train_auc: %.3f, train_loss: %.3f, test_auc: %.3f, test_loss: %.3f'
                     % (epoch+1, args.num_epochs, train_auc, train_loss, test_auc, test_loss))

        if best_auc < test_auc:
            torch.save(model.state_dict(), f'{args.output_dir}/{args.fasta_file}/model.pkl')
            best_auc = test_auc

            ### ROC curve ###
            fpr, tpr, _ = roc_curve(train_gt, train_pred)
            plt.plot(fpr, tpr, label='training ROC curve')
            fpr, tpr, _ = roc_curve(test_gt, test_pred)
            plt.plot(fpr, tpr, label='test ROC curve')
            plt.title(f'Training AUC={train_auc:.3f}, Test AUC={test_auc:.3f}')
            plt.xlabel('FPR: False positive rate')
            plt.ylabel('TPR: True positive rate')
            plt.legend()
            plt.grid()
            plt.savefig(f'{args.output_dir}/{args.fasta_file}/roc_curve.png', dpi=300)
            plt.close()

            ### Prediction ###
            df_train = pd.DataFrame(data=train_pred,
                                    columns=['prediction'], 
                                    index=train_name)
            df_train.to_csv(f'{args.output_dir}/{args.fasta_file}/2ndDL_prediction_train.csv')

            df_test = pd.DataFrame(data=test_pred,
                                    columns=['prediction'], 
                                    index=test_name)
            df_test.to_csv(f'{args.output_dir}/{args.fasta_file}/2ndDL_prediction_test.csv')
     
    time_end = time()
    print('exec time: %d s, best_auc: %.3f' % (time_end-time_start, best_auc))


