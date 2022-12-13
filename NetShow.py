from torchsummary import summary
import torch

import Teacher
import Teacher1
import Teacher2
import Student
import Student1
import Student2

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

teacher = Teacher.Teacher()  
student = Student.Student()
teacher1 = Teacher1.Teacher() 
student1 = Student1.Student() 
teacher2 = Teacher2.Teacher() 
student2 = Student2.Student()

nets = [teacher,student,teacher1,student1,teacher2,student2]
for net in nets:
    print("0")
    net = net.to(device)
    summary(net,(1,28,28))
    print()
