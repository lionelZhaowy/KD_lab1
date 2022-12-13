from torch import nn

class Teacher(nn.Module):
    # 教师模型先定义 三个隐藏层fc1，fc2，fc3
    def __init__(self,in_channels=1,num_classes=10):
        super(Teacher, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,1200)
        self.fc2 = nn.Linear(1200,1200)
        self.fc3 = nn.Linear(1200,num_classes)
        self.dropout = nn.Dropout(p = 0.5) # 使用dropout防止过拟合

    def forward(self,x):
        x = x.view(-1,784)
        x = self.fc1(x)
        x = self.relu(x) # 前向传播使用线性整流relu激活函数

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x
