import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class inverse_model(nn.Module):

    def __init__(self):
        super(inverse_model, self).__init__()

        # 59*2, 113*2
        self.inputsize = 226

        # 15
        self.outputsize = 11

        self.fc1 = nn.Linear(self.inputsize, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.outputsize)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = self.bn2(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = torch.sigmoid(self.fc3(x))

        return x

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)

if __name__ == "__main__":
    import time
    model = inverse_model()
    model.eval()
    with torch.no_grad():
        input_data = torch.ones((1, 226))
        run_times = 1000
        t0 = time.time()
        for i in range(run_times):
            output_data = model.forward(input_data)
            # print(output_data.shape)
        t1 = time.time()
        print(1/((t1-t0)/run_times))
        print(((t1-t0)/run_times))
