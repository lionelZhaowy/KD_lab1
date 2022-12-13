import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import transforms

import numpy as np
import time
import copy

# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from Teacher import Teacher
from Student import Student

#调用GPU训练
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def KD(teacher,device,train_loader,test_loader,T,alpha):
    #列表用于绘图
    # acc_train = []
    acc_test = []
    
    teacher = teacher.to(device)
    teacher.eval()
    student = Student()
    student = student.to(device)
    #启用batch normalization & dropout
    student.train()
    #定义蒸馏损失函数的两项
    hard_loss = nn.CrossEntropyLoss()
    hard_loss = hard_loss.to(device)
    soft_loss = nn.KLDivLoss(reduction='batchmean')
    soft_loss = soft_loss.to(device)
    #设置优化器
    optimizer = torch.optim.Adam(student.parameters(),lr = 1e-4)
    
    epochs = 20

    #开始蒸馏
    max_acc = 0
    for epoch in range(epochs):
        file_path = './log/kd_T{}_alpha{}.txt'.format(T,alpha)
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
            distillation_loss = soft_loss(
                F.log_softmax(student_preds/T,dim = 1),
                F.softmax(teacher_preds/T,dim = 1)
                )
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
            #训练精度
            # for data,target in tqdm(train_loader):
            #     data = data.to(device)
            #     target = target.to(device)
            #     preds = student(data)
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
        print("\n"+"-"*15+"\n")
        f.write("\n"+"-"*15+"\n")
        #启用batch normalization & dropout
        student.train()

    print("训练结束，最优精度:{}\n".format(max_acc),"-"*15)
    f.write("训练结束，最优精度:{}\n".format(max_acc))
    f.close()
    torch.save(best_model,'D:/Anaconda3/program_files/KD_lab/models/kd_T{}_alpha{}.pth'.format(T,alpha))

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
        plt.savefig('./fig/kd_T{}_alpha{}.png'.format(T,alpha))
        plt.clf()
        # plt.show()
    #返回本次T、alpha条件下的模型最优精度
    return max_acc

if __name__ == '__main__':
    #设置随机种子确保每一次运行结果相同
    torch.cuda.manual_seed(0)
    #是否调用GPU进行训练
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    #使用cudnn对训练进行加速
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #加载数据集
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
    #加载Teacher模型
    teacher_model_path = "D:/Anaconda3/program_files/KD_lab/models/Teacher.pth"
    teacher = Teacher()
    teacher.load_state_dict(torch.load(teacher_model_path))
    #进行知识蒸馏
    acc_list = []
    for T in range(0,10):
        acc_list.append([])
        for alpha in range(0,10):
            acc = KD(teacher,device,train_loader,test_loader,T+1,alpha/10)
            acc_list[T].append(acc)
    print("训练结束\nacc_list = ")
    print(acc_list)

    #绘图:acc-T-alpha
    T = list(range(0,10))
    alpha = np.arange(0,1,0.1)
    # T = list(range(0,2))
    # alpha = np.arange(0,0.2,0.1)

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
