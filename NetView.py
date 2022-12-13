# 针对有网络模型，但还没有训练保存 .pth 文件的情况
import netron

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

nets = [teacher,student,teacher1,student1,teacher2,student2]
paths = ['./models/Teacher.pth','./models/Student.pth',
    './models1/Teacher.pth','./models1/Student.pth',
    './models2/Teacher.pth','./models2/Student.pth']
for net,path in zip(nets,paths):
    netron.start(path)  # 输出网络结构