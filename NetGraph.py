import torch
from torchviz import make_dot

import Teacher
import Teacher1
import Teacher2
import Student
import Student1
import Student2

teacher = Teacher.Teacher()  
student = Student.Student()
teacher1 = Teacher1.Teacher() 
student1 = Student1.Student() 
teacher2 = Teacher2.Teacher() 
student2 = Student2.Student()

def var_name(var,all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]

x=torch.rand(8,1,28,28)
nets = [teacher,student,teacher1,student1,teacher2,student2]
for net in nets:
    y = net(x)
    g = make_dot(y)
    g.render('./{}'.format(var_name(net)), view=True) 


# 这三种方式都可以
# g = make_dot(y)
# g=make_dot(y, params=dict(model.named_parameters()))
# g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))

# 这两种方法都可以
# g.view() # 会生成一个 Digraph.gv.pdf 的PDF文件
# 会自动保存为一个 espnet.pdf，
# 第二个参数为True,则会自动打开该PDF文件，为False则不打开
# g.render('espnet_model', view=False) 