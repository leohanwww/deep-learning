激活函数

y = b + w1x1 + w2x2 信号汇总
y = h(b + w1x1 + w2x2) h称为激活函数

a = b + w1x1 +w2x2
y = h(a)

阶跃函数
import numpy as np

def step_function(x):
	y = x > 0 
	return y.astype(np.int)
x = np.array([-1.0, 1.0, 2.0])
y = x > 0
>>> y
array([False,  True,  True], dtype=bool)
y.astype(np.int) #将bool型转换成int型
>>>y
array([0, 1, 1])

sigmoid函数
import matplotlib.pyplot as plt
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
#这个函数支持numpy数组的计算
>>> x = np.array([-1.0, 1.0, 2.0])
>>> sigmoid(x)
array([ 0.26894142, 0.73105858, 0.88079708])
#画出图形
>>>x = np.arange(-5.0, 5.0 0.1)
>>>y = sigmoid(x)
>>>plt.plot(x, y)
>>>plt.ylim(-0.1, 1.1)
>>>plt.show()

ReLu函数
ReLU函数在输入大于0 时，直接输出该值；在输入小于等于0 时，输出0

def ReLu(x):
	return np.maximum(0, x)
	









