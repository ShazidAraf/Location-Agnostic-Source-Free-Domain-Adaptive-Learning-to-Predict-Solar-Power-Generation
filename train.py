import argparse, os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from model.model import Solar_Model
from dataloader.data_loader import Solar_Loader
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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--ckp', type=str, default='./checkpoint')
    parser.add_argument('--data_path', type=str, default= '/home/araf/Desktop/Solar_power/data')
    parser.add_argument('--city_name', type=str, default= 'CA')
    parser.add_argument('--n_fold', type=int, default = 5)
    parser.add_argument('--resume', type=int, default = 0)
    
    args = parser.parse_args()

    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

def train(epoch, loader, model, optimizer):



    loader = tqdm(loader)
    model.train()

    for i, sample in enumerate(loader):

        input_1 = sample[0]
        label = sample[1]
        # label = torch.unsqueeze(label,dim=1)


        model.zero_grad()
        optimizer.zero_grad()
 

        input_1 = (input_1.type(torch.FloatTensor)).to(device)
        label = (label.type(torch.LongTensor)).to(device)

        out = model(input_1)
        # out_2 = torch.nn.functional.softmax(out, dim=1)
        # out_3 = torch.argmax(out_2)
        # pdb.set_trace()

        loss = criterion(out,label)
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch}; Loss: {loss.item():.5f}; '
                f'lr: {lr:.5f}'
            )
        )

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


    # model = Solar_Model(10,5)
    model = Solar_Model(6,5)

    if args.resume:
        model.load_state_dict(torch.load('{0}/best_ckp_{1}.pt'.format(args.ckp,args.city_name)))
        print("loading pretrained model from {0}/best_ckp_{1}.pt".format(args.ckp,args.city_name))
    else:
        print("Traing from scratch")
        model.apply(initialize_weights)

    model = model.to(device)
    # model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001 )

    all_train_loss = []
    all_val_acc = []

    highest_acc = 0

    for i in range(args.epoch):


        train_data = Solar_Loader(data_dir = args.data_path,city_name = args.city_name,split='train')
        train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=2)

        val_data = Solar_Loader(data_dir = args.data_path,city_name = args.city_name,split='test')
        val_loader = DataLoader(val_data, batch_size = val_data.__len__(), shuffle=True, num_workers=2)

        train_loss = train(i, train_loader, model, optimizer)
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
            torch.save(model.state_dict(), os.path.join(args.ckp,'best_ckp_{0}.pt'.format(args.city_name)))

        all_train_loss.append(train_loss)
        all_val_acc.append(val_acc)
        index = [l for l in range(i+1)]


        plt.plot(index,all_train_loss , 'r')
        plt.savefig('{0}/all_loss_train_src_{1}.png'.format(args.ckp,args.city_name))
        plt.figure().clear()

        np.save('{0}/all_acc_train_src_{1}.npy'.format(args.ckp,args.city_name), np.array(all_val_acc))


        plt.plot(index,all_val_acc , 'b')
        plt.savefig('{0}/all_acc_train_src_{1}.png'.format(args.ckp,args.city_name))
        plt.figure().clear()



        plt.plot(range(gt.shape[0]),gt,'r*')
        plt.plot(range(gt.shape[0]),pred,'b*')
        plt.savefig('{0}/pred_compare.png'.format(args.ckp))
        plt.figure().clear()


