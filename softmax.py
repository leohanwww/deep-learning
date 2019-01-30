import numpy as np

a = np.array([0.3, 2.9, 4.0])

def softmax(a):
	exp_a = np.exp(a)
	sum_a = np.sum(exp_a)
	y = exp_a / sum_a
	return y
#softmax函数其实是求结果（概率值）的函数，把结果变成小数
def softmax_c(a):#增加常量，防止溢出
	c = np.max(a)
	exp_a = np.exp(a - c)
	sum_a = np.sum(exp_a)
	y = exp_a / sum_a
	return y
	
>>> a = np.array([0.3, 2.9, 4.0])
>>> y = softmax(a)
>>> print(y)
[ 0.01821127 0.24519181 0.73659691]
>>> np.sum(y)
1.0
y[0]的概率是0.018（1.8%），y[1]的概率
是0.245（24.5%），y[2]的概率是0.737（73.7%）
一般而言，神经网络只把输出值最大的神经元所对应的类别作为识别结果




























