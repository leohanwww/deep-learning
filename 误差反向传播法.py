计算损失函数关于权重参数的梯度，使用数组微分在计算上比较费时

根据计算图从左到右传播为正向传播
可以通过反向传播高效计算导数

计算图的反向传播
链式法则
z = t**2
t = x + y
aZ/aX = aZ/aZ * aZ/aT * aT/aX
反向传播即z关于x的导数aZ/aX
这样的好处就是只要关心局部计算，可以计算出x和y有微小变化时z做出何种变化

加法节点的反向传播
aL/aZ ......aL/az * 1  
       ......aL/aZ * 1
加法节点反向传播原封不动地将右边的值传递到左边，两个加数各获得一份

乘法节点的反向传播
乘法的反向传播会将上游的值乘以正向传播时的输入信号的“翻转值”
后传递给下游
z = x * y
aL/aZ ......aL/aZ * Y
       ......aL/aY * X

乘法层的实现
class MulLayer:
	def __init__(self, x, y):
		self.x = None
		self.y = None
		
	def forward(self, x, y):
		self.x = x
		self.y = y
		out = x * y
		return out

	def backward(self, dout):
		dx = dout * self.y
		dy = dout * self.x
		return dx, dy

apple = 100
apple_num = 2
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

#backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)


加法层的实现

class AddLayer:
	def __init__(self):
		pass
	
	def forward(self, x, y):
		out = x + y
		return out
		
	def backward(self, dout):
		dx = dout * 1
		dy = dout * 1
		return dx, dy


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num) #(1)
orange_price = mul_orange_layer.forward(orange, orange_num) #(2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price) #(3)
price = mul_tax_layer.forward(all_price, tax) #(4)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) #(4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) #(3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price) #(2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #(1)

print(price) # 715
print(dapple_num, dapple, dorange, dorange_num, dtax) # 110 2.2 3.3 165 650

激活函数层的实现

def Relu:
	def __init__(self):
		self.mask = None
#mask是由True/False构成的数组，它会把正向传播时的输入x的元素中小于等于0 的地方保存为True，其
#他地方（大于0 的元素）保存为False。
	def forward(self, x):
		self.mask = (x <= 0 )
		out = x.copy()
		out[self.mask] = 0 #在mask为True的地方，out为0
		return out #这样的out矩阵可以让正值通过，负值为0即不通过

	def backword(self, out):
		dout[self.mask] = 0
		dx = dout
		return dx

>>> x = np.array( [[1.0, -0.5], [-2.0, 3.0]] )
>>> print(x)
[[ 1. -0.5]
[-2. 3. ]]
>>> mask = (x <= 0)
>>> print(mask)
[[False True]
[ True False]]


sigmoid层

class Sigmoid:
	def __init__(self):
		self.out = None
		
	def forward(self, x):
		out = 1 / (1 + np.exp(-x))
		self.out = out
		return out

	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out
		return dx
#正向传播时将输出保存在了实例变量out中。然后，反向
#传播时，使用该变量out进行计算。

Affine层-访射层

>>> X = np.random.rand(2) # 输入
>>> W = np.random.rand(2,3) # 权重
>>> B = np.random.rand(3) # 偏置
>>>
>>> X.shape # (2,)
>>> W.shape # (2, 3)
>>> B.shape # (3,)
>>>
>>> Y = np.dot(X, W) + B
>>> y
array([-1.086231  ,  0.67445112,  0.68213472])
这里，X、W、B分别是形状为(2,)、(2, 3)、(3,) 的多维数组。这样一
来，神经元的加权和可以用Y = np.dot(X, W) + B计算出来。然后，Y 经过
激活函数转换后，传递给下一层。这就是神经网络正向传播的流程

Affine层的反向传播
aL/aX = aL/aY * W(T)
aL/aW = x(T) * aL/aY
w(T)表示W的转制，w为（2，3）的转制就为（3，2）

批Affine层
>>> X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
>>> B = np.array([1, 2, 3])
>>>
>>> X_dot_W
array([[ 0, 0, 0],
[ 10, 10, 10]])
>>> X_dot_W + B
array([[ 1, 2, 3],
[11, 12, 13]])
正向传播时，偏置会被加到每一个数据（第1 个、第2 个……）上
因此，反向传播时，各个数据的反向传播的值需要汇总为偏置的元素
>>> dY = np.array([[1, 2, 3,], [4, 5, 6]])
>>> dY
array([[1, 2, 3],
[4, 5, 6]])
>>>
>>> dB = np.sum(dY, axis=0)
>>> dB
array([5, 7, 9])

class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None
		
	def forward(self, x):
		self.x = x
		out = np.dot(x, self.w) + self.b
		return out
		
	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T dout)
		self.db = np.sum(dout, axis=0)
		return dx

softmax_with_loss层

class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None
		
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)
		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size
		return dx

反向两层网络的实现

params 保存神经网络的参数的字典型变量。
params['W1']是第1 层的权重，params['b1']是第1 层的偏置。
params['W2']是第2 层的权重，params['b2']是第2 层的偏置
layers 保存神经网络的层的有序字典型变量。
以layers['Affine1']、layers['ReLu1']、layers['Affine2']的形式，
通过有序字典保存各个层
lastLayer 神经网络的最后一层。
本例中为SoftmaxWithLoss层

__init__(self, input_size,
hidden_size, output_size,
weight_init_std)
进行初始化。
参数从头开始依次是输入层的神经元数、隐藏层的
神经元数、输出层的神经元数、初始化权重时的高
斯分布的规模
predict(self, x) 进行识别（推理）。
参数x是图像数据
loss(self, x, t) 计算损失函数的值。
参数X是图像数据、t是正确解标签
accuracy(self, x, t) 计算识别精度
numerical_gradient(self, x, t) 通过数值微分计算关于权重参数的梯度（同上一章）
gradient(self, x, t) 通过误差反向传播法计算关于权重参数的梯度


import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		#初始化权重参数
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

		#生成层
		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu1'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.lastLayer = SoftmaxWithLoss()

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		if t.ndim != 1 : t = np.argmax(t, axis=1)
		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy

	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)
		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
		return grads

	def gradient(self, x, t):
		# forward
		self.loss(x, t)
		# backward
		dout = 1
		dout = self.lastLayer.backward(dout)
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)

		# 设定
		grads = {}
		grads['W1'] = self.layers['Affine1'].dW
		grads['b1'] = self.layers['Affine1'].db
		grads['W2'] = self.layers['Affine2'].dW
		grads['b2'] = self.layers['Affine2'].db
		return grads


















