# nmt_course
题目：

  在NiuTensor中实现0-7的数字异或。
  
看看：
输入为1*6的tensor,输出为1*8的tensor。其中输入为两个进行异或值的三位二进制数，输出为8位的one-hot向量。
前向传播：
h1=w1*x
h2=w1*x+b
h3=hardtanh(h2)
output=softmax(h3)
反向传播:
使用交叉熵函数计算loss.
dE/dw1= dE/doutput * doutpu

