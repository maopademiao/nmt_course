# 题目：

  在NiuTensor中实现0-7的数字异或。
  
# 设计思路：

  用分类的思想做。<br>
  输入为 64x6 的tensor,输出为 64x8 的tensor。其中输入为两个进行异或值的三位二进制数，输出为 1x8 的one-hot向量。<br>
  激活函数用sigmoid。<br>
  损失用交叉熵计算。<br>
  对结果取argmax, 概率最大的作为异或的结果。

### 输入输出

* 输入为两个三位二进制数进行拼接后的1X6矩阵,如1和2进行异或的输入为[001010]<br>
* 输出为1x8矩阵，因为0~7进行异或只有8种输出，如1和2异或的结果为3，对应one-hot输出为[00010000]<br>

### 参数设置及初始化

* 权重W 范围[-0.99f,0.99f]<br>
* 偏置b 初始化为0<br>
* epoch 1500<br>
* 学习率 0.005f,每隔400轮降为原来的90%<br>
  
### 前向传播：

 *  h1=w1*x<br>
 *  h2=w1*x+b<br>
 *  h3=sigmoid(h2)<br>
 *  output=softmax(h3)
  
### 反向传播:

  损失函数使用交叉熵函数计算loss.<br>
  * dE/dw1 = dE/doutput * doutput/dh3 * dh3/dh2 * dh2/dw1<br>
  * dE/db = dE/doutput * doutput/dh3 * dh3/dh2 * dh2/db
  
### 参数更新：

  * w1 = w1 - dE/dw1 * learning_rate<br>
  * b = b - dE/db * learning_rate

# 结果

* 使用八分类的方法，在单层隐藏层的网络下正确率三分之一。<br>

# 问题

* loss不能收敛到1<br>
* 多加隐藏层结果并没有变好。
