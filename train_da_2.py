# python train_da.py --src_city_name CA --tgt_city_name NY

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
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--ckp', type=str, default='./checkpoint')
    parser.add_argument('--data_path', type=str, default= '/home/araf/Desktop/Solar_power/data')
    parser.add_argument('--src_city_name', type=str, default= 'CA')
    parser.add_argument('--tgt_city_name', type=str, default= 'NY')
    parser.add_argument('--n_fold', type=int, default = 5)
    
    args = parser.parse_args()

    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()



def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)



def train(epoch,tgt_loader, model, optimizer):


    model.train()

    for i, sample_tgt in tqdm(enumerate(tgt_loader)):

        input_tgt = sample_tgt[0]
        label_tgt = sample_tgt[1]

        model.zero_grad()
        optimizer.zero_grad()
 
        input_tgt = (input_tgt.type(torch.FloatTensor)).to(device)
        label_tgt = (label_tgt.type(torch.LongTensor)).to(device)

        out_tgt = model(input_tgt)
        loss = criterion(out_tgt,label_tgt)
        
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']


    return loss





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


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        # print('Initialize Conv1D')
        torch.nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        # print('Initialize Batch Normalization')
        torch.nn.init.constant_(m.weight.data, 1)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        # print('Initialize Batch Linear Layer')
        torch.nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':

    args = parse_args()
    print(args)

    os.makedirs(args.ckp,exist_ok=True)


    model = Solar_Model(6,5)
    model.load_state_dict(torch.load('{0}/best_ckp_{1}.pt'.format(args.ckp,args.src_city_name)))
    print("loading pretrained model from {0}/best_ckp_{1}.pt".format(args.ckp,args.src_city_name))

    print("Source data : {0}, Adapting For {1}".format(args.src_city_name,args.tgt_city_name))

    model = model.to(device)
    
    model = configure_model(model)
    # model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001 )

    all_train_loss = []
    all_val_acc = []

    highest_acc = 0

    for i in range(args.epoch):

        train_data_tgt = Solar_Loader(data_dir = args.data_path,city_name = args.tgt_city_name,split='train')
        train_loader_tgt = DataLoader(train_data_tgt, batch_size = args.batch_size, shuffle=True, num_workers=2)

        val_data = Solar_Loader(data_dir = args.data_path,city_name = args.tgt_city_name,split='test')
        val_loader = DataLoader(val_data, batch_size = val_data.__len__(), shuffle=True, num_workers=2)

        train_loss = train(i, train_loader_src,train_loader_tgt, model, optimizer)
        train_loss = train_loss.detach().numpy()

        val_acc,pred,gt = evaluate(val_loader, model)
        # val_acc = val_acc.detach().numpy()
        print('val acc: ', val_acc)
        # pred = pred.detach().numpy()
        # gt = gt.detach().numpy()
        n = gt.shape[0]



        # import pdb

        # pdb.set_trace()

        if val_acc>highest_acc:
            highest_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.ckp,'DA_best_ckp_{0}_{1}.pt'.format(args.src_city_name,args.tgt_city_name)))

        all_train_loss.append(train_loss)
        all_val_acc.append(val_acc)
        index = [l for l in range(i+1)]


        plt.plot(index,all_train_loss , 'r')
        plt.savefig('{0}/all_loss_DA_src_{1}_tgt_{2}.png'.format(args.ckp,args.src_city_name,args.tgt_city_name))
        plt.figure().clear()

        plt.plot(index,all_val_acc , 'b')
        plt.savefig('{0}/all_acc_DA_src_{1}_tgt_{2}.png'.format(args.ckp,args.src_city_name,args.tgt_city_name))
        plt.figure().clear()

        np.save('{0}/all_acc_DA_src_{1}_tgt_{2}.npy'.format(args.ckp,args.src_city_name,args.tgt_city_name), np.array(all_val_acc))

        plt.plot(range(gt.shape[0]),gt, 'r*')
        plt.plot(range(gt.shape[0]),pred, 'b*')
        plt.savefig('{0}/pred_compare.png'.format(args.ckp))
        plt.figure().clear()
