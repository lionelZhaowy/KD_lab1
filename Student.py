from torch import nn

class Student(nn.Module):
    def __init__(self,in_channel = 1,num_classes = 10):
        super(Student,self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,num_classes)
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self,x):
        x = x.view(-1,784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x