import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

import time
import copy

from Teacher import Teacher

#列表用于绘图
acc_train = []
acc_test = []

#设置种子保证每次运行结果相同
torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
#数据集
train_data = torchvision.datasets.MNIST(
    root = "D:/Anaconda3/datasets/",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
#训练集
test_data = torchvision.datasets.MNIST(
    root = "D:/Anaconda3/datasets/",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)
#构建数据加载器
train_loader = DataLoader(dataset=train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=False)
#教师例化
model = Teacher()
model = model.to(device)
best_model = None
#损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)

epochs = 20

#写入文件
file_path = './log/teacher.txt'
with open(file_path,"w") as f:
    max_acc = 0
    for epoch in range(epochs):
        print("运行时间:%s\tepoch:%d"%(time.strftime("%Y-%m-%d %H:%M:%S"),epoch+1))
        f.write("运行时间:%s\tepoch:%d\n"%(time.strftime("%Y-%m-%d %H:%M:%S"),epoch+1))
        start_time = time.time()
        #训练
        #启用batch normalization & dropout
        model.train()
        for data,target in tqdm(train_loader):
            data = data.to(device)
            target = target.to(device)
            preds = model(data)
            #计算损失
            loss = criterion(preds,target)
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #参数优化
            optimizer.step()
        end_time = time.time()
        print("本阶段损失CrossEntropyLoss:%f"%(loss))
        f.write("本阶段损失CrossEntropyLoss:%f\n"%(loss))
        print("本阶段用时:%fs"%(round(end_time-start_time,7)))
        f.write("本阶段用时:%fs\n"%(round(end_time-start_time,7)))
        #关闭batch normalization & dropout
        model.eval()
        num_correct = 0
        num_samples = 0
        #测试
        with torch.no_grad():
            # #训练精度
            # for data,target in tqdm(train_loader):
            #     data = data.to(device)
            #     target = target.to(device)
            #     preds = model(data)
            #     predictions = preds.max(1).indices
            #     num_correct += (predictions.eq(target)).sum().item()
            #     num_samples += predictions.size(0)
            # acc = num_correct/num_samples
            # acc_train.append(round(acc,5))
            # print(("Epoch:{}\t 训练Accuracy:{:5f}").format(epoch+1,acc))
            # f.write(("Epoch:{}\t 训练Accuracy:{:5f}\n").format(epoch+1,acc))
             #在测试集测试
            
            for data,target in tqdm(test_loader):
                data = data.to(device)
                target = target.to(device)
                preds = model(data)
                predictions = preds.max(1).indices
                num_correct += (predictions.eq(target)).sum().item()
                num_samples += predictions.size(0)
            acc = num_correct/num_samples
            acc_test.append(round(acc,5))

            #保存测试集上最优精度的模型
            if acc > max_acc:
                print("最优精度模型更换:{}->{}".format(acc,max_acc))
                f.write("最优精度模型更换:{}->{}".format(acc,max_acc))
                max_acc = acc
                best_model = copy.deepcopy(model.state_dict())
            print(("Epoch:{}\t 测试Accuracy:{:5f}").format(epoch+1,acc))
            f.write(("Epoch:{}\t 测试Accuracy:{:5f}\n").format(epoch+1,acc))
        print("\n"+"-"*15+"\n")
        f.write("\n"+"-"*15+"\n")
        #启用batch normalization & dropout
        model.train()
    print("训练结束，最优精度:{}".format(max_acc))
    f.write("训练结束，最优精度:{}\n".format(max_acc))
    torch.save(best_model,'D:/Anaconda3/program_files/KD_lab/models/Teacher.pth')

x = list(range(epochs))
#绘制精度折线
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("Teacher:acc_train-epoch")
# plt.plot(x,acc_train,"g",marker='D',label="train")
plt.plot(x,acc_test,"r",marker='*',label="test")

# for a,b in zip(x,acc_train):
#     plt.text(a,b,b,ha='center',va='bottom',fontsize=8)
for a,b in zip(x,acc_test):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=8)

plt.legend()
plt.savefig('./fig/teacher.png')
plt.show()