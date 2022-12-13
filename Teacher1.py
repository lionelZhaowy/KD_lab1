from torch import nn
import torch.nn.functional as F

#273k个参数，文件大小约为26.89MB

class Teacher(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(Teacher,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
        )
        self.pool = nn.MaxPool2d(
            kernel_size=(2,2),
            stride=(2,2)
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1)
        )
        self.fc1 = nn.Linear(256*7*7,num_classes)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x