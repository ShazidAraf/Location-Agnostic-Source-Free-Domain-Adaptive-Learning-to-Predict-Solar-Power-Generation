import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pdb



# Define the neural network model
class Solar_Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Solar_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.flatten = nn.Flatten()


        self.fc1 = nn.Linear(128 * 1, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):


        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":


    model = Solar_Model(15,5)
    x = torch.rand(100,15,1)

    pdb.set_trace()
    z = model(x)

    # pdb.set_trace()
    print(z.shape)
