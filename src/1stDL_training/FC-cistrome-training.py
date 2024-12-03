import numpy as np
import pickle
import os
import argparse
from glob import glob
from tqdm import tqdm
from time import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Dataset, dna2onehot, fix_seed, get_FC_3layer

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import logging


def load_data(TF, dataset_path):
    print('load %s ...' % TF)
    posi = glob('%s/posi/*%s*' % (dataset_path, TF))[0]
    nega = glob('%s/nega/*%s*' % (dataset_path, TF))[0]

    X = []
    Y = []
    f = open(posi, "r")
    line = f.readline()
    while line:
        line2 = line.rstrip()
        if len(line2) == args.length:
            OneHotArr = dna2onehot(list(line2))
            X.append(OneHotArr)
            Y.append(1)
        line = f.readline()

    f = open(nega, "r")
    line = f.readline()
    while line:
        line2 = line.rstrip()
        if len(line2) == args.length:
            OneHotArr = dna2onehot(list(line2))
            X.append(OneHotArr)
            Y.append(0)
        line = f.readline()

    X = np.array(X)
    Y = np.array(Y)

    print('X size: ' + str(X.shape))
    print('Y size: ' + str(Y.shape))

    return X, Y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--dataset_path', default='./dataset/',
                        type=str, help='File path to posi/nega dataset')
    parser.add_argument('-len', '--length',default=31,
                        type=int, help='Length for input size')
    parser.add_argument('-d', '--device', default='cuda:0',
                        type=str, help='GPU device')
    parser.add_argument('-epochs', '--num_epochs',
                        default=1000, type=str, help='number of epochs')
    parser.add_argument('-p', '--patience', default=5,
                        type=str, help='patience of early stopping')
    parser.add_argument('-bs', '--batch_size', default=512,
                        type=str, help='batch size')
    parser.add_argument('-lr', '--lr', default=3e-4,
                        type=float, help='learning rate')
    parser.add_argument('-o', '--output_path',
                        default='./model/', type=str, help="output file name")
    args = parser.parse_args()

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.output_path+'training.log')
    logging.basicConfig(level=logging.DEBUG, handlers=[
                        stream_handler, file_handler])

    ######## training for all TF #########
    # with open('./TF_dict.pkl', 'rb') as tf:
    #     TF_dict = pickle.load(tf)
    
    TF_dict = open(args.dataset_path+'TF_dict.txt', 'r').read().split('\n')

    # for ID, TF_name in TF_dict.items():
    for ID, TF_name in enumerate(TF_dict):
        time_start = time()
        fix_seed(seed=0)

        logging.info('=================================')
        logging.info('%d: %s' % (ID, TF_name))

        ############ load dataset ##############
        X, Y = load_data(TF_name, args.dataset_path)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.33, random_state=111)

        ############ define model & loader & optimizer #############
        # model = get_FC_3layer().to(args.device)
        model = get_FC_3layer(args.length).to(args.device)

        weight = 1 / np.eye(2)[Y].sum(axis=0)
        weight /= weight.sum()
        weight = torch.tensor(weight).float().to(args.device)
        loss_function = nn.CrossEntropyLoss(weight=weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_dataset = Dataset(X_train, Y_train)
        test_dataset = Dataset(X_test, Y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        ############ training #############
        best_auc = 0
        for epoch in range(args.num_epochs):
            ####################################
            model.train()

            losses = []
            gt, pred = [], []

            for data, label in train_loader:
                data, label = data.to(args.device), label.to(args.device)
                y = model(data)
                prob = F.softmax(y, dim=1)
                loss = loss_function(y, label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.item())
                gt.extend(label.cpu().detach().numpy())
                pred.extend(prob[:, 1].cpu().detach().numpy())

            train_loss = np.array(losses).mean()
            train_auc = roc_auc_score(gt, pred)

            ####################################
            model.eval()

            losses = []
            gt, pred = [], []

            with torch.no_grad():
                for data, label in test_loader:
                    data, label = data.to(args.device), label.to(args.device)
                    y = model(data)
                    prob = F.softmax(y, dim=1)
                    loss = loss_function(y, label)

                    losses.append(loss.item())
                    gt.extend(label.cpu().detach().numpy())
                    pred.extend(prob[:, 1].cpu().detach().numpy())

            test_loss = np.array(losses).mean()
            test_auc = roc_auc_score(gt, pred)

            logging.info('[%d/%d]: train_auc: %.2f, train_loss: %.4f, test_auc: %.2f, test_loss: %.4f'
                         % (epoch+1, args.num_epochs, train_auc*100, train_loss, test_auc*100, test_loss))

            if best_auc < test_auc:
                torch.save(model.state_dict(), '%s%s.pkl' %
                           (args.output_path, TF_name))
                best_auc = test_auc
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping == args.patience:
                    break

        time_end = time()
        logging.info('exec time: %d s, best_auc: %.2f' %
                     (time_end-time_start, best_auc*100))
