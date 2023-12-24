import argparse, os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from model.model import * 
from dataloader.data_loader import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pdb
import copy

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--ckp', type=str, default='./checkpoint')
    parser.add_argument('--data_path', type=str, default= '/home/araf/Desktop/Weather/data')
    parser.add_argument('--city_name', type=str, default= 'CA')
    parser.add_argument('--n_fold', type=int, default = 5)
    
    args = parser.parse_args()

    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def calculate_accuracy(pred,gt):

    N_acc = np.sum(pred==gt)
    N = gt.shape[0]

    return N_acc, N




def evaluate(loader, model,plot=0):



    loader = tqdm(loader)
    N_acc  = 0
    N = 0

    all_gt = []
    all_pred  = []


    model.eval()

    for i, sample in enumerate(loader):

        input_1 = sample[0]
        label = sample[1]
        # label = torch.unsqueeze(label,dim=1)


        input_1 = (input_1.type(torch.FloatTensor)).to(device)
        label = (label.type(torch.LongTensor)).to(device)

        out = model(input_1)
        out_2 = torch.nn.functional.softmax(out, dim=1)
        out_3 = torch.argmax(out_2, dim=1)

        # loss = criterion(out,label)
        all_gt+=(label.tolist())
        all_pred+=(out_3.tolist())


        # print('Val loss : ',loss)
    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)
    N_acc,N = calculate_accuracy(all_pred,all_gt)


    Acc = N_acc/N

    return Acc,all_pred, all_gt


if __name__ == '__main__':

    args = parse_args()
    print(args)

    os.makedirs(args.ckp,exist_ok=True)


    model = Solar_Model(6,5)
    model.load_state_dict(torch.load('{0}/best_ckp_{1}.pt'.format(args.ckp,args.city_name)))

    print("loading pretrained model from {0}/best_ckp_{1}.pt".format(args.ckp,args.city_name))



    model = model.to(device)
    # model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001 )



    test_data = Solar_Loader(data_dir = args.data_path,city_name = args.city_name,split='test')
    test_loader = DataLoader(test_data, batch_size = test_data.__len__(), shuffle=True, num_workers=2)


    val_acc,pred,gt = evaluate(test_loader, model)
    val_acc = val_acc.detach().numpy()
    print("Acc: ",val_acc)

    pred = pred.detach().numpy()
    gt = gt.detach().numpy()
    n = gt.shape[0]


    plt.plot(range(gt.shape[0]),gt, 'r')
    plt.plot(range(gt.shape[0]),pred, 'b')
    plt.savefig('{0}/test_pred_compare_{1}.png'.format(args.ckp,args.city_name))
    plt.figure().clear()
