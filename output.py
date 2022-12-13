import matplotlib.pyplot as plt
import numpy as np

def get_max(array):
    max_num,max_row,max_col = 0,0,0
    for row in range(10):
        l = array[row]
        for col in range(10):
            if l[col] > max_num:
                max_num = l[col]
                max_row = row
                max_col = col
    return max_num,max_row,max_col
                
def show_3d(X,Y,z_list):
    Z = []
    for i in z_list:
        Z.append(np.array(i))
    Z = np.array(Z)

    print(X)
    print(Y)
    print(Z)

    # fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis', edgecolor='none')
    ax3.set_title('acc-T&alpha')
    plt.show()

#T恒定,alpha变化
def show_2d_T(alpha_list,T,index):
    x_list = np.arange(0,1,0.1)
    if index == 1:
        teacher = [0.977400 for i in range(10)]
        student = [0.943600 for i in range(10)]
    elif index == 0:
        teacher = [0.983100 for i in range(10)]
        student = [0.885800 for i in range(10)]
    elif index == 2:
        teacher = [0.981900 for i in range(10)]
        student = [0.952400 for i in range(10)]
    plt.xlabel("alpha")
    plt.ylabel("acc")
    plt.title("kd_alpha_T{}_{}".format(T,index))
    plt.plot(x_list,alpha_list,"r",marker='*',label="kd")
    plt.plot(x_list,student,"g",marker='o',label="student")
    plt.plot(x_list,teacher,"b",marker='o',label="teacher")

    for a,b in zip(x_list,alpha_list):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=5)
    for a,b in zip(x_list,teacher):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=5)
    for a,b in zip(x_list,student):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=5)

    #绘制图像
    plt.legend()
    #保存图像，路径改成自己电脑的就行！！！！！！
    plt.savefig('./test{}/acc-alpha_T{}.png'.format(index,T,index))
    #显示图像
    # plt.show()
    #清空绘图
    plt.clf()


#alpha恒定,T变化
def show_2d_alpha(T_list,alpha,index):
    x_list = list(range(0,10))
    if index == 1:
        teacher = [0.977400 for i in range(10)]
        student = [0.943600 for i in range(10)]
    elif index == 0:
        teacher = [0.983100 for i in range(10)]
        student = [0.885800 for i in range(10)]
    elif index == 2:
        teacher = [0.981900 for i in range(10)]
        student = [0.952400 for i in range(10)]
    plt.xlabel("T")
    plt.ylabel("acc")
    plt.title("kd_T_alpha{}_{}".format(alpha,index))
    plt.plot(x_list,T_list,"r",marker='*',label="kd")
    plt.plot(x_list,student,"g",marker='o',label="student")
    plt.plot(x_list,teacher,"b",marker='o',label="teacher")

    for a,b in zip(x_list,T_list):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=5)
    for a,b in zip(x_list,teacher):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=5)
    for a,b in zip(x_list,student):
        plt.text(a,b,b,ha='center',va='bottom',fontsize=5)

    #绘制图像
    plt.legend()
    #保存图像，路径改成自己电脑的就行！！！！！！
    plt.savefig('./test{1}/acc-T_alpha{0}.png'.format(alpha,index))
    #显示图像
    # plt.show()
    #清空绘图
    plt.clf()


if __name__ == '__main__':
    T = list(range(0,10))
    alpha = np.arange(0,1,0.1)
    X,Y = np.meshgrid(T,alpha)

    #模型训练精度列表
    acc_train0 = [[0.8896, 0.9048 ,0.8781 ,0.8936 ,0.8969 ,0.8923 ,0.8899 ,0.9036 ,0.8989 ,0.8904],
                [0.8927, 0.8642 ,0.8848 ,0.8768 ,0.8866 ,0.8879 ,0.8973 ,0.8974 ,0.878  ,0.8894],
                [0.8906, 0.8785 ,0.8754 ,0.8862 ,0.8969 ,0.8884 ,0.8991 ,0.8865 ,0.8955 ,0.886 ],
                [0.8594, 0.8876 ,0.8882 ,0.8795 ,0.876  ,0.8872 ,0.8745 ,0.8792 ,0.888  ,0.8905],
                [0.8731, 0.8866 ,0.8879 ,0.8904 ,0.8915 ,0.889  ,0.8917 ,0.8943 ,0.893  ,0.8907],
                [0.8826, 0.8815 ,0.8899 ,0.8818 ,0.8918 ,0.899  ,0.8981 ,0.8915 ,0.8748 ,0.8998],
                [0.8487, 0.878  ,0.893  ,0.8973 ,0.8948 ,0.8821 ,0.8891 ,0.892  ,0.8989 ,0.8976],
                [0.8777, 0.8752 ,0.8932 ,0.8928 ,0.8931 ,0.8876 ,0.8998 ,0.8924 ,0.8929 ,0.8958],
                [0.84 ,  0.8626 ,0.8942 ,0.8902 ,0.887  ,0.8907 ,0.8923 ,0.8968 ,0.8925 ,0.9022],
                [0.87 ,  0.8899 ,0.8721 ,0.8846 ,0.8992 ,0.8901 ,0.898  ,0.892  ,0.8945 ,0.9018]]

    acc_train1 = [[0.9744, 0.9736, 0.9764, 0.9794, 0.9782, 0.9774, 0.9791, 0.9785, 0.98, 0.9805], 
                [0.9745, 0.9747, 0.9765, 0.9767, 0.9773, 0.9781, 0.9805, 0.9815, 0.9805, 0.9801], 
                [0.9752, 0.977,  0.9786, 0.9779, 0.9785, 0.9771, 0.978, 0.98  ,  0.9808, 0.98], 
                [0.9728, 0.9769, 0.976, 0.9797, 0.9804, 0.9813, 0.9797, 0.9784, 0.98, 0.9813], 
                [0.9729, 0.9774, 0.9776, 0.9801, 0.9791, 0.9792, 0.9801, 0.9815, 0.9814, 0.979], 
                [0.9751, 0.9785, 0.9798, 0.9789, 0.9783, 0.9797, 0.982, 0.9797, 0.9793, 0.9813], 
                [0.9736, 0.976, 0.9776, 0.9772, 0.9794, 0.9787, 0.9788, 0.9797, 0.9795, 0.9818], 
                [0.9736, 0.9781, 0.9784, 0.9787, 0.9785, 0.981, 0.9823, 0.9801, 0.979, 0.9819], 
                [0.9732, 0.9798, 0.9795, 0.9808, 0.9795, 0.9809, 0.9807, 0.9786, 0.98, 0.9819], 
                [0.973, 0.9789, 0.9786, 0.9794, 0.9796, 0.9805, 0.98, 0.9783, 0.982, 0.9785]]   
    
    acc_train2 = [[0.9821, 0.9854, 0.9866, 0.9891, 0.9887, 0.9904, 0.9919, 0.9921, 0.993, 0.9944],
                [0.9819, 0.986, 0.9887, 0.9914, 0.992, 0.9933, 0.9937, 0.993, 0.9926, 0.9924], 
                [0.9805, 0.9876, 0.9904, 0.9922, 0.9927, 0.9934, 0.9935, 0.9936, 0.9935, 0.9933], 
                [0.9796, 0.9889, 0.9912, 0.9925, 0.9919, 0.9932, 0.9943, 0.9932, 0.9928, 0.9929],
                [0.9809, 0.9905, 0.9918, 0.9931, 0.9925, 0.994, 0.9935, 0.9922, 0.9931, 0.9936], 
                [0.9799, 0.991, 0.993, 0.9928, 0.9926, 0.9926, 0.9935, 0.9921, 0.9919, 0.9933], 
                [0.9798, 0.9918, 0.9927, 0.9935, 0.9927, 0.9925, 0.9928, 0.9924, 0.9929, 0.9927],
                [0.9774, 0.9927, 0.9932, 0.9935, 0.993, 0.9925, 0.9919, 0.9925, 0.9922, 0.9934], 
                [0.9794, 0.9938, 0.9931, 0.993, 0.993, 0.9921, 0.9923, 0.9929, 0.993, 0.9931], 
                [0.9799, 0.9932, 0.9932, 0.9928, 0.9941, 0.9933, 0.9937, 0.993, 0.9926, 0.9917]]

    max_num0,max_row0,max_col0 = get_max(acc_train0)
    max_num1,max_row1,max_col1 = get_max(acc_train1)
    max_num2,max_row2,max_col2 = get_max(acc_train2)
    print("组一:\n蒸馏最大精度出现在T={0},alpha={1}处\n最高精度为{2},提高了{3}\n".format(max_row0+1,max_col0/10,max_num0,max_num0-0.885800))
    print("组二:\n蒸馏最大精度出现在T={0},alpha={1}处\n最高精度为{2},提高了{3}\n".format(max_row1+1,max_col1/10,max_num1,max_num1-0.943600))
    print("组三:\n蒸馏最大精度出现在T={0},alpha={1}处\n最高精度为{2},提高了{3}\n".format(max_row2+1,max_col2/10,max_num2,max_num2-0.952400))
    
    #显示3D图像
    show_3d(X,Y,acc_train0)
    show_3d(X,Y,acc_train1)
    show_3d(X,Y,acc_train2)

    # 显示T恒定，alpha变化的10中情况
    for T in range(10):
        show_2d_T(acc_train0[T],T,0)
        show_2d_T(acc_train1[T],T,1)
        show_2d_T(acc_train2[T],T,2)
    #显示alpha恒定，T变化的10中情况
    for i in range(10):
        a = [x[i] for x in acc_train0]
        show_2d_alpha(a,i/10,0)
    
    for i in range(10):
        a = [x[i] for x in acc_train1]
        show_2d_alpha(a,i/10,1)
        
    for i in range(10):
        a = [x[i] for x in acc_train2]
        show_2d_alpha(a,i/10,2)