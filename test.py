import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from Teacher2 import Teacher
from Student2 import Student

import numpy as np
import time

def Test(device,model_name,model_path,test_loader,T=1,alpha=1):
    #加载模型
    if model_name == 'teacher':
        model = Teacher()
    elif model_name == 'student':
        model = Student()
    elif model_name == 'kd':
        model = Student()
    else:
        print("未找到该网络模型!!!")
        return
    #模型初始化
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    file_path = './test2/log/{}_test_.txt'.format(model_name)

    with open(file_path,"a") as f:
        print("运行时间:%s"%(time.strftime("%Y-%m-%d %H:%M:%S")))
        f.write("运行时间:%s\n"%(time.strftime("%Y-%m-%d %H:%M:%S")))
        if model_name == 'kd':
            print("T = {}\talpha = {}".format(T,alpha))
            f.write("T = {}\talpha = {}\n".format(T,alpha))
        num_correct = 0
        num_samples = 0
        #测试开始
        with torch.no_grad():
            for data,target in tqdm(test_loader):
                data = data.to(device)
                target = target.to(device)
                preds = model(data)
                predictions = preds.max(1).indices
                num_correct += (predictions.eq(target)).sum().item()
                num_samples += predictions.size(0)
            acc = num_correct/num_samples

            print(("测试精度:{:5f}").format(acc))
            f.write(("测试精度:{:5f}\n").format(acc))
        print("-"*30)
        f.write("-"*30+"\n")

    return acc


if __name__ == '__main__':
    #是否调用GPU
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    #设置随机种子
    # torch.cuda.manual_seed(0)
    #使用cudnn加速
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    #加载数据集
    test_data = torchvision.datasets.MNIST(
        root='D:/Anaconda3/datasets/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    #创建数据加载器
    test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True)

    #模型测试
    #教师测试
    Teacher_path = './models2/Teacher.pth'
    Test(device,'teacher',Teacher_path,test_loader)
    #学生测试
    Student_path = './models2/Student.pth'
    Test(device,'student',Student_path,test_loader)
    #蒸馏效果测试
    kd_list = []
    for T in range(0,10):
        kd_list.append([])
        # for alpha in range(0,10,1):
        for alpha in np.arange(0,1,0.1):
            KD_path = './models2/kd_T{}_alpha{}.pth'.format(T+1,alpha)
            acc = Test(device,'kd',KD_path,test_loader,T+1,alpha)
            kd_list[T].append(acc)
    print("kd_list = ")
    print(kd_list)
