# nmt_course
##用分类的方法实现异或。<br>
###输入为1*6的tensor,输出为1*8的tensor。其中输入为两个进行异或值的三位二进制数，输出为8位的one-hot向量。<br>
###前向传播：<br>
h1=w1*x<br>
h2=w1*x+b<br>
h3=hardtanh(h2)<br>
output=softmax(h3)<br>
###反向传播:<br>
使用交叉熵函数计算loss.
dE/dw1= dE/doutput * doutpu

