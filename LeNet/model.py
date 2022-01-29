import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) #in_channals, out_channals, kernal_size, 默认步幅为1
        self.pool1 = nn.MaxPool2d(2, 2) #参数是kernal_size和stride 只改变h和w
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(32, 3, 32, 32) output(32, 16, 28, 28), 通过卷积改变了channal
        x = self.pool1(x)            # output(16, 14, 14) b h w
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5) ，设置为-1，表示第一维的维度会自动推理
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10) ，这里不需要添加softmax层了
        return x

import torch
input1 = torch.rand([32, 3, 32, 32]) #N C H W
model = LeNet()
print(model)
output = model(input1)