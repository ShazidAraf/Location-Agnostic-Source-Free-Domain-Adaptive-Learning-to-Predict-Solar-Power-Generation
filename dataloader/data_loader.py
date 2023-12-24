import os
import torch

import torch.nn.functional  as F
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset,DataLoader
import numpy as np
import copy
import pdb
import pandas as pd
from sklearn.preprocessing import StandardScaler




class Solar_Loader(Dataset):
    
    def __init__(self,data_dir,city_name,split):

        self.data_dir = data_dir
        self.city_name = city_name
        self.split = split

        # pdb.set_trace()
        selected_idx = np.array(list(set(range(15)) - set(list(range(5))+[8,10,11,14])))

        # selected_idx = np.array(list(set(range(15)) - set(list(range(5)))))
        # print(selected_idx)


        label_idx = 15
        
        # load data

        train_data = np.load('{0}/{1}/train.npy'.format(data_dir,city_name),allow_pickle=True)
        test_data = np.load('{0}/{1}/test.npy'.format(data_dir,city_name),allow_pickle=True)

        X_train_raw = train_data[:,selected_idx]
        X_test_raw = test_data[:,selected_idx]
        
        y_train = train_data[:,label_idx]
        y_test = test_data[:,label_idx]
 

         # Standardizing data

        scaler = StandardScaler()
        scaler.fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        self.data_bin = [0,8,16,24,32,40]

        if city_name == 'FL':
            self.data_bin = [0.0, 7.90, 15.79, 23.69, 31.59, 39.48]
        elif city_name == 'NY':
            self.data_bin = [0.0, 8.0, 15.99, 23.99, 31.99, 39.98]
        elif city_name == 'CA':
            self.data_bin = [0.0, 9.66, 19.31, 28.97, 38.63, 48.28]

        if self.split=='train':
            self.X = X_train
            self.Y = y_train

        elif self.split=='test':
            self.X = X_test
            self.Y = y_test

        # print(self.X.shape,self.Y.shape)

    def __len__(self):

        return self.X.shape[0]

    def __getitem__(self, idx):
        
        x = self.X[idx]
        x = np.expand_dims(x,axis=1)
        y = self.Y[idx]

        if self.data_bin is not None:
            n_bin = len(self.data_bin) - 1 

            # Categorizing Target Variable
            for i in range(len(self.data_bin)-1):
                if self.data_bin[i]<=y and self.data_bin[i+1]>y:
                    break

            class_label = i

                # if 20>=y>=0:
                #     y=1
                # elif 40>=y>=21:
                #     y=2
                # elif 60>=y>=41:
                #     y=3
                # elif 80>=y>=61:
                #     y=4
                # elif 100>=y>=81:
                #     y=5
                # elif 120>=y>=101:
                #     y=6
                # elif 140>=y>=121:
                #     y=7
                # elif 162>=y>=141:
                #     y=8
        
        # print(x.shape)
        return x,class_label
        
