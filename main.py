import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
# from torchinfo import summary
from tqdm import tqdm

from Teacher2 import Teacher
from Student2 import Student

import numpy as np
import matplotlib.pyplot as plt
import time
import copy


#模型训练函数
def train_teacher(device,train_loader,test_loader,epochs):
    #模型例化
    model = Teacher()
    model = model.to(device)
    #损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    #优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    #训练循环
    max_acc = 0
    acc_test = []
    best_model = None
    file_path = './log2/Teacher.txt'
    with open(file_path,"w") as f:
        for epoch in range(epochs):
            print("运行时间:%s\tepoch:%d"%(time.strftime("%Y-%m-%d %H:%M:%S"),epoch+1))
            f.write("运行时间:%s\tepoch:%d\n"%(time.strftime("%Y-%m-%d %H:%M:%S"),epoch+1))
            start_time = time.time()
            #训练开始
            model.train()
            for data,target in tqdm(train_loader):
                data = data.to(device)
                target = target.to(device)
                preds = model(data)
                #计算sunsh
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
            #评估
            model.eval()
            num_correct = 0
            num_samples = 0

            with torch.no_grad():
                for data.target in tqdm(test_loader):
                    data = data.to(device)
                    target = target.to(device)
                    preds = model(data)
                    predictions = preds.max(1).indices
                    num_correct += (predictions.eq(target)).sum().item()
                    num_samples += predictions.size(0)
                acc = num_correct/num_samples
                acc_test.append(round(acc,5))

                print("epoch:{}\t测试精度:{:.5f}".format(epoch+1,acc))
                f.write("epoch:{}\t测试精度:{:.5f}\n".format(epoch+1,acc))
                #保存在测试集上精度最优的模型
                if acc > max_acc:
                    print("最优精度模型更换:{}->{}\n".format(acc,max_acc))
                    f.write("最优精度模型更换:{}->{}\n".format(acc,max_acc))
                    max_acc = acc
                    best_model = copy.deepcopy(model.state_dict())
                print("-"*30)
                f.write("-"*30)
            model.train()
        print("训练结束，最优精度:{}".format(max_acc))
        f.write("训练结束，最优精度:{}\n".format(max_acc))
        torch.save(best_model,'./models2/Teacher.pth')
    x = list(range(epochs))
    #绘制精度折线
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Teacher2:acc_train-epoch")
    plt.plot(x,acc_test,"r",marker='*',label="test")

    for a,b in zip(x,acc_test):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=8)

    plt.legend()
    plt.savefig('./fig2/Teacher.png')
    plt.show()


def train_student(device,train_loader,test_loader,epochs):
    #模型例化
    model = Student()
    model = model.to(device)
    #损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    #优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    #训练循环
    max_acc = 0
    acc_test = []
    best_model = None
    file_path = './log2/Student.txt'
    with open(file_path,"w") as f:
        for epoch in range(epochs):
            print("运行时间:%s\tepoch:%d"%(time.strftime("%Y-%m-%d %H:%M:%S"),epoch+1))
            f.write("运行时间:%s\tepoch:%d\n"%(time.strftime("%Y-%m-%d %H:%M:%S"),epoch+1))
            start_time = time.time()
            #训练开始
            model.train()
            for data,target in tqdm(train_loader):
                data = data.to(device)
                target = target.to(device)
                preds = model(data)
                #计算sunsh
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
            #评估
            model.eval()
            num_correct = 0
            num_samples = 0

            with torch.no_grad():
                for data.target in tqdm(test_loader):
                    data = data.to(device)
                    target = target.to(device)
                    preds = model(data)
                    predictions = preds.max(1).indices
                    num_correct += (predictions.eq(target)).sum().item()
                    num_samples += predictions.size(0)
                acc = num_correct/num_samples
                acc_test.append(round(acc,5))

                print("epoch:{}\t测试精度:{:.5f}".format(epoch+1,acc))
                f.write("epoch:{}\t测试精度:{:.5f}\n".format(epoch+1,acc))
                #保存在测试集上精度最优的模型
                if acc > max_acc:
                    print("最优精度模型更换:{}->{}\n".format(acc,max_acc))
                    f.write("最优精度模型更换:{}->{}\n".format(acc,max_acc))
                    max_acc = acc
                    best_model = copy.deepcopy(model.state_dict())
                print("-"*30)
                f.write("-"*30)
            model.train()
        print("训练结束，最优精度:{}".format(max_acc))
        f.write("训练结束，最优精度:{}\n".format(max_acc))
        torch.save(best_model,'./models2/Student.pth')
    x = list(range(epochs))
    #绘制精度折线
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Student2:acc_train-epoch")
    plt.plot(x,acc_test,"r",marker='*',label="test")

    for a,b in zip(x,acc_test):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=8)

    plt.legend()
    plt.savefig('./fig2/Student.png')
    plt.show()


def KD(device,train_loader,test_loader,epochs,T,alpha):
    #声明列表用于绘图
    acc_test = []

    #教师模型加载
    teacher_model_path = "./models2/Teacher.pth"
    teacher = Teacher()
    teacher.load_state_dict(torch.load(teacher_model_path))
    teacher = teacher.to(device)
    teacher.eval()
    #学生模型定义
    student = Student()
    student = student.to(device)
    student.train()

    #蒸馏损失定义
    hard_loss = nn.CrossEntropyLoss()
    hard_loss = hard_loss.to(device)
    soft_loss = nn.KLDivLoss(reduction='batchmean')
    soft_loss = soft_loss.to(device)
    #设置优化器
    optimizer = torch.optim.Adam(student.parameters(),lr = 1e-4)

    #开始蒸馏
    max_acc = 0
    for epoch in range(epochs):
        file_path = './log2/kd_T{}_alpha{}.txt'.format(T,alpha)
        f = open(file_path,"w")
        print("运行时间:%s\nepoch:%d\tT = %d\talpha = %f"%(time.strftime("%Y-%m-%d %H:%M:%S"),epoch+1,T,alpha))
        f.write("运行时间:%s\nepoch:%d\tT = %d\talpha = %f\n"%(time.strftime("%Y-%m-%d %H:%M:%S"),epoch+1,T,alpha))
        start_time = time.time()
        #训练
        for data,target in tqdm(train_loader):
            data = data.to(device)
            target = target.to(device)
            #教师模型产生soft target
            with torch.no_grad():
                teacher_preds = teacher(data)
            #学生模型预测
            student_preds = student(data)
            #student与真实标签产生hard loss
            student_loss = hard_loss(student_preds,target)
            #student与teacher产生的平滑分布产生soft loss
            distillation_loss = soft_loss(F.log_softmax(student_preds/T,dim = 1),F.softmax(teacher_preds/T,dim = 1))
            #总体蒸馏损失
            loss = alpha*student_loss + (1-alpha)*distillation_loss
            #清零梯度
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #参数优化
            optimizer.step()
        end_time = time.time()

        print("本阶段损失hard_loss:%f\tsoft_loss:%f\tloss:%f"%(student_loss,distillation_loss,loss))
        f.write("本阶段损失hard_loss:%f\tsoft_loss:%f\tloss:%f\n"%(student_loss,distillation_loss,loss))
        print("本阶段用时:%fs"%(round(end_time-start_time,7)))
        f.write("本阶段用时:%fs\n"%(round(end_time-start_time,7)))
        
        #关闭batch normalization & dropout
        student.eval()
        num_correct = 0
        num_samples = 0
        #测试
        with torch.no_grad():
            #在测试集测试
            for data,target in tqdm(test_loader):
                data = data.to(device)
                target = target.to(device)
                preds = student(data)
                predictions = preds.max(1).indices
                num_correct += (predictions.eq(target)).sum().item()
                num_samples += predictions.size(0)
            acc = num_correct/num_samples
            acc_test.append(round(acc,5))
            #保存测试集上最优精度的模型
            if acc > max_acc:
                print("最优精度模型更换:{}->{}".format(acc,max_acc))
                max_acc = acc
                best_model = copy.deepcopy(student.state_dict())
            print(("Epoch:{}\t 测试Accuracy:{:5f}").format(epoch+1,acc))
            f.write(("Epoch:{}\t 测试Accuracy:{:5f}\n").format(epoch+1,acc))
        print("-"*30)
        f.write("-"*30)
        #启用batch normalization & dropout
        student.train()

    print("训练结束，最优精度:{}\n".format(max_acc),"-"*15)
    f.write("训练结束，最优精度:{}\n".format(max_acc))
    f.close()
    torch.save(best_model,'./models2/kd_T{}_alpha{}.pth'.format(T,alpha))

    if(len(acc_test) == epochs):
        x = list(range(epochs))
        #绘制精度折线
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.title("kd_T{}_alpha{}".format(T,alpha))
        # plt.plot(x,acc_train,"g",marker='D',label="train")
        plt.plot(x,acc_test,"r",marker='*',label="test")

        # for a,b in zip(x,acc_train):
        #     plt.text(a,b,b,ha='center',va='bottom',fontsize=8)
        for a,b in zip(x,acc_test):
            plt.text(a,b,b,ha='center',va='bottom',fontsize=8)

        plt.legend()
        plt.savefig('./fig2/kd_T{}_alpha{}.png'.format(T,alpha))
        plt.clf()
        # plt.show()
    #返回本次T、alpha条件下的模型最优精度
    return max_acc


if __name__ == '__main__':
    #训练参数声明
    epochs = 20
    #是否调用GPU
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    #设置随机种子
    torch.cuda.manual_seed(0)
    #使用cudnn加速
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    #加载数据集
    train_data = torchvision.datasets.MNIST(
        root='D:/Anaconda3/datasets/',
        train = True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_data = torchvision.datasets.MNIST(
        root='D:/Anaconda3/datasets/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    #创建数据加载器
    train_loader = DataLoader(dataset=train_data,batch_size=64,shuffle=True)
    test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True)

    #教师模型训练
    train_teacher(device,train_loader,test_loader,epochs)
    #对照组:学生模型训练
    train_student(device,train_loader,test_loader,epochs)
    #实验组:学生模型知识蒸馏
    acc_list = []
    for T in range(0,10):
        acc_list.append([])
        for alpha in np.arange(0,1,0.1):
            acc = KD(device,train_loader,test_loader,epochs,T+1,alpha)
            acc_list[T].append(acc)
    print("训练结束\nacc_list = ")
    print(acc_list)

    #绘图:acc-T-alpha
    T = list(range(0,10))
    alpha = np.arange(0,1,0.1)

    X,Y = np.meshgrid(T,alpha)
    # #将列表转化为array
    Z = []
    for i in acc_list:
        Z.append(np.array(i))
    Z = np.array(Z)
    print(X)
    print(Y)
    print(Z)

    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis', edgecolor='none')
    ax3.set_title('acc-T&alpha')
    plt.show()
