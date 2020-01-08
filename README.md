# nmt_course
## 题目：

  在NiuTensor中实现0-7的数字异或。
  
## 设计思路：

  用分类的思想做。<br>
  每次输入为 1x6 的tensor,输出为 1x8 的tensor。其中输入为两个进行异或值的三位二进制数，输出为 1x8 的one-hot向量。<br>
  激活函数用sigmoid。<br>
  损失用交叉熵计算。<br>
  对结果取argmax, 概率最大的作为异或的结果。

### 输入输出

### 参数初始化
  
### 前向传播：

  h1=w1*x<br>
  h2=w1*x+b<br>
  h3=sigmoid(h2)<br>
  output=softmax(h3)
  
### 反向传播:

  使用交叉熵函数计算loss.<br>
  dE/dw1= dE/doutput * doutput
  

### 参数更新：



