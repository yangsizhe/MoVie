import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformNet_STN_PerImage(nn.Module):
    def __init__(self, num_channels):
        super(TransformNet_STN_PerImage, self).__init__()
        self.num_channals = num_channels
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x = x.view(-1, 3, x.size()[-2], x.size()[-1])
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x.view(-1, self.num_channals, 224, 224)
    
class TransformNet_STN(nn.Module):
    def __init__(self, num_channels):
        super(TransformNet_STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=7, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # import pdb;pdb.set_trace()
        # print("====",theta)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class TransformNet_STN1(nn.Module):
    def __init__(self, num_channels):
        super(TransformNet_STN1, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=7, stride=2),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, stride=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class TransformNet_STN2(nn.Module):
    def __init__(self, num_channels):
        super(TransformNet_STN2, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(16, 10, kernel_size=3),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
