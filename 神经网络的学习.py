神经网络的学习
让神经网络从数据中学习，自动调整权重参数

在计算机视觉领域，常用的特征量包括
SIFT、SURF和HOG等。使用这些特征量将图像数据转换为向量，然后对
转换后的向量使用机器学习中的SVM、KNN等分类器进行学习

泛化能力是指处理未被观察过的数据（不包含在训练数据中的数据）的
能力。获得泛化能力是机器学习的最终目标。比如，在识别手写数字的问题
中，泛化能力可能会被用在自动读取明信片的邮政编码的系统上。此时，手
写数字识别就必须具备较高的识别“某个人”写的字的能力

损失函数是表示神经网络性能的“恶劣程度”的指标，即当前的
神经网络对监督数据在多大程度上不拟合，在多大程度上不一致

经典损失函数

>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]#有softmax得出，可以解释为概率
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
将正确解标签表示为1，其他标签表示为0 的表示方法称为one-hot 表示

import numpy as np

def mean_squared_error(y, t):#均方误差函数
#均方误差会计算神经网络的输出和正确解监督数据的各个元素之差的平方，再求总和
	return 0.5 * sum((y - t) ** 2)
	
>>> # 设“2”为正确解
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
>>>
>>> # 例1：“2”的概率最高的情况（0.6）
>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
>>> mean_squared_error(np.array(y), np.array(t))
0.097500000000000031
>>>
>>> # 例2：“7”的概率最高的情况（0.6）
>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
>>> mean_squared_error(np.array(y), np.array(t))
0.59750000000000003


def cross_entropy_error(y, t):#交叉嫡误差函数
#交叉熵误差的值是由正确解标签所对应的输出结果决定的
	delta = 1e-7
	return -np.sum(t * np.log(y + delta))
	
这里，参数y和t是NumPy数组。函数内部在计算np.log时，加上了一
个微小值delta。这是因为，当出现np.log(0)时，np.log(0)会变为负无限大
的-inf，这样一来就会导致后续计算无法进行。作为保护性对策，添加一个
微小值可以防止负无限大的发生
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
>>> cross_entropy_error(np.array(y), np.array(t))
0.51082545709933802
>>>
>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
>>> cross_entropy_error(np.array(y), np.array(t))
2.3025840929945458

在进行神经网络的学习时，不能将识别精度作为指标。因为如果以
识别精度为指标，则参数的导数在绝大多数地方都会变为0

mini-batch学习
从全部数据中选出一部分，作为全部数据的“近似”。神经网络的学习也是从训练数据中选出一批数据（称为mini-batch, 小批量），然后对每个mini-batch 进行学习。比如，从60000 个训练数据中随机
选择100 笔，再用这100 笔数据进行学习。这种学习方式称为mini-batch 学习。

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)
#随机抽取10笔数据
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

mini-batch版交叉嫡误差函数
def cross_entropy_error(y, t):
	if y.ndim == 1:
	#y为神经网络的输出，y的维度为1时，即求单个数据的交叉嫡误差
		t = t.reshape(1, t.size)#改变形状,变成多维数组
		y = y.reshape(1, y.size)#变成y.shape(1, 3)
	
	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + 1e-7)) / batch_size

#当监督数据是标签形式（非one-hot 表示，而是像“2”“7”这样的标签）时
def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#如果y不是一维的，用np.arange(batch_size)生成一个从0 到batch_size-1的数组
#比如当batch_size为5时，np.arange(batch_size) 会生成一个NumPy 数组[0, 1, 2, 3, 4]
#因为t中标签是以[2, 7, 0, 9, 4]的形式存储的，所以y[np.arange(batch_size),t]能抽出各个数据的正确解标签对应的神经网络的输出
#（在这个例子中，y[np.range(batch_size), t] 会生成NumPy 数组[y[0,2], y[1,7], y[2,0],y[3,9], y[4,4]]）。

导数
“导数”在神经网络学习中的作用:在神经网络的学习中，寻找最优参数（权重和偏置）时，
要寻找使损失函数的值尽可能小的参数。为了找到使损失函数的值尽可能小
的地方，需要计算参数的导数（确切地讲是梯度），然后以这个导数为指引，
逐步更新参数的值

假设有一个神经网络，现在我们来关注这个神经网络中的某一个权重参
数。此时，对该权重参数的损失函数求导，表示的是“如果稍微改变这个权
重参数的值，损失函数的值会如何变化”。如果导数的值为负，通过使该权
重参数向正方向改变，可以减小损失函数的值；反过来，如果导数的值为正，
则通过使该权重参数向负方向改变，可以减小损失函数的值。不过，当导数
的值为0 时，无论权重参数向哪个方向变化，损失函数的值都不会改变，此
时该权重参数的更新会停在此处。
我们对权重参数求导！以获得改变的方向和大小！

导数就是表示某个瞬间的变化量
def numerical_diff(f, x):#求导函数
	h = 1e-4#h取很小的值
	return (f(x + h) - f(x)) / (2 * h)
#利用微小的差分求导数的过程称为数值微分

def numerical_diff(f, x):#中心差分法，
	h = 1e-4
	return (f(x+h) - f(x-h)) / (2*h)
	
	
def function_1(x):
	return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt
x = np.arange(0, 20 ,0.1)
y =  function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

>>> numerical_diff(function_1, 5)#计算function_1在x=5处的导数，也就是斜率
0.1999999999990898
>>> numerical_diff(function_1, 10)
0.2999999999986347

偏导数
偏导数和单变量的导数一样，都是求某个地方的斜率。不过，
偏导数需要将多个变量中的某一个变量定为目标变量，并将其他变量固定为
某个值
def function_2(x):
	return x[0]**2 + x[1]**2
#或者return np.sum(x**2)

def function_tmp1(x0):#固定第一个值
	return x0*x0 + 4.0**2.0

def function_tmp2(x1):#固定第二个值
	return 3.0**2.0 + x1*x1
	
>>> numerical_diff(function_tmp1, 3.0)#这个函数只能有一个参数，所以
#上面的函数要确定一个x1的值3.0，求得关于x0的导数
6.00000000000378
>>> numerical_diff(function_tmp2, 4.0)
7.999999999999119
偏导数和单变量的导数一样，都是求某个地方的斜率

梯度
def numerical_gradient(f, x):#计算梯度
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x) # 生成和x形状相同的所有元素为0的数组
	
	for idx in range(x.size):
		tmp_val = x[idx]
		# f(x+h)的计算
		x[idx] = tmp_val + h
		fxh1 = f(x)
		
		# f(x-h)的计算
		x[idx] = tmp_val - h
		fxh2 = f(x)
		
		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val # 还原值
		
	return grad

#求各点（x0,x1）的梯度
>>> numerical_gradient(function_2, np.array([3.0, 4.0]))
array([ 6., 8.])
>>> numerical_gradient(function_2, np.array([0.0, 2.0]))
array([ 0., 4.])
>>> numerical_gradient(function_2, np.array([3.0, 0.0]))
array([ 6., 0.])
梯度表示的是各点处的函数值减小最多的方向
#通过巧妙地使用梯度来寻找函数最小值（或者尽可能小的值）的方法就是梯度法
在梯度法中，函数的取值从当前位置沿着梯度方向前进一定距离，然后在新的地方重新求梯度，再沿着新梯度方向前进，
如此反复，不断地沿梯度方向前进。像这样，通过不断地沿梯度方向前进，
逐渐减小函数值的过程就是梯度法（gradient method）

def gradient_descent(f, init_x, lr=0.01, step_num=100):
#梯度下降法
#lr是学习率，这里是指定的一个确定值，但是一般是动态的
	x = init_x
	
	for i in range(step_num)
		grad = numerical_gradient(f, x)
		x = lr * grad
		
	return x
f是要进行最优化的函数，init_x 是初始值，lr 是学习率learning
rate，step_num 是梯度法的重复次数。
numerical_gradient(f,x) 会求函数的
梯度，用该梯度乘以学习率得到的值进行更新操作

用梯度法求f(x0+x1)=x0**2+x1**2的最小值
>>> def function_2(x):
... return x[0]**2 + x[1]**2
...
>>> init_x = np.array([-3.0, 4.0])
>>> gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
array([ -6.11110793e-10, 8.14814391e-10])

神经网络的梯度
#神经网络的梯度指的是损失函数关于权重参数的梯度
w = w11 w12 w13 #权重参数
	w21 w22 w23
#aL是损失函数
aL/aw = aL/aw11 aL/aw12 aL/aw13 #这就是梯度，形状和w相同
		aL/aw21 aL/aw22 aL/aw23
aL/aw11表示当权重参数w11 稍微变化时，损失函数L会发生多大变化

import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
	def __init__(self):
		self.W = np.random.randn(2,3) # 用高斯分布进行初始化，获得随机权重

	def predict(self, x):#预测结果
		return np.dot(x, self.W)

	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y, t)
		return loss
>>> net = simpleNet()
>>> print(net.W) # 权重参数
[[ 0.47355232 0.9977393 0.84668094],
[ 0.85557411 0.03563661 0.69422093]])
>>>
>>> x = np.array([0.6, 0.9])
>>> p = net.predict(x)
>>> print(p)
[ 1.05414809 0.63071653 1.1328074]
>>> np.argmax(p) # 最大值的索引
2
>>>
>>> t = np.array([0, 0, 1]) # 正确解标签
>>> net.loss(x, t)
0.92806853663411326

>>> dW = numerical_gradient(net.loss(x,t), net.W)#求loss(x,t)关于权重W的梯度
>>> print(dW)#aL/aW
[[ 0.21924763 0.14356247 -0.36281009]
[ 0.32887144 0.2153437 -0.54421514]]
numerical_gradient(f, net.W)的结果是dW，一个形状为2 × 3 的二维数组。
aL/aw11的值为0.21，这表示如果将w11 增加h，那么损失函数的值会增加0.2h
aL/aw23值为-0.54，这表示如果将w23 增加h，损失函数的值将减小0.5h，
因此，从减小损失函数值的观点来看，w23 应向正方向更新，w11 应向负方向更新。至于
更新的程度，w23 比w11的贡献要大

学习算法的实现步骤
前提
神经网络存在合适的权重和偏置，调整权重和偏置以便拟合训练数据的
过程称为“学习”。神经网络的学习分成下面4 个步骤。
步骤1（mini-batch）
从训练数据中随机选出一部分数据，这部分数据称为mini-batch。我们
的目标是减小mini-batch 的损失函数的值。
步骤2（计算梯度）
为了减小mini-batch 的损失函数的值，需要求出各个权重参数的梯度。
梯度表示损失函数的值减小最多的方向。
步骤3（更新参数）
将权重参数沿梯度方向进行微小更新。
步骤4（重复）
重复步骤1、步骤2、步骤3。

import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

	def __init__(self, intput_size, hidden_size, output_size, weight_init_std=0.01):
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
		self.params['b2'] = np.zeros(output_size)

	def predict(self, x):
		W1, W2 =  self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']
		a1 = np.dot(x,W1) + b1
		z1 = sigmod(a1)
		a2 = np.dot(z1,W2) + b2
		y = softmax(a2)
		return y

	def loss(self,x,t):
		y = self.predict(x)
		return cross_entropy_error(y,t)

	def numerical_gradient(self,x,t):
		loss_W = lambda W : self.loss(x,t)
		grads = {}#保存各个参数的梯度
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
		return grads
实例化：
x = np.random.rand(100, 784) # 伪输入数据（100笔）
y = net.predict(x)

x = np.random.rand(100, 784) # 伪输入数据（100笔）
t = np.random.rand(100, 10) # 伪正确解标签（100笔）
grads = net.numerical_gradient(x, t) # 计算梯度
grads['W1'].shape # (784, 100)
grads['b1'].shape # (100,)
grads['W2'].shape # (100, 10)
grads['b2'].shape # (10,)


mini-batch的实现

(x_train, t_train), (x_test, t_test) = \ load_mnist(normalize=True, one_hot_
laobel = True)

train_loss_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
	# 获取mini-batch
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]
	# 计算梯度
	grad = network.numerical_gradient(x_batch, t_batch)
	# grad = network.gradient(x_batch, t_batch) # 高速版!
	# 更新参数
	for key in ('W1', 'b1', 'W2', 'b2'):
	network.params[key] -= learning_rate * grad[key]
	# 记录学习过程
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

以上的是基于训练数据的损失值，严格地讲是“对训练数据的某
个mini-batch 的损失函数”的值，神经网络学习的最初目标是掌握泛化能力，因此，要评价神经网络的泛
化能力，就必须使用不包含在训练数据中的数据

epoch是一个单位。一个epoch表示学习中所有训练数据均被使用过
一次时的更新次数。比如，对于10000 笔训练数据，用大小为100
笔数据的mini-batch 进行学习时，重复随机梯度下降法100 次，所
有的训练数据就都被“看过”了A。此时，100次就是一个epoch。

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \ load_mnist(normalize=True, one_hot_
laobel = True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

# 超参数
iters_num = 10000
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
	# 获取mini-batch
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	# 计算梯度
	grad = network.numerical_gradient(x_batch, t_batch)
	# grad = network.gradient(x_batch, t_batch) # 高速版!

	# 更新参数
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	# 计算每个epoch的识别精度
	if i % iter_per_epoch == 0:#是epoch的整数倍的话，即第N次学习过所有数据
		train_acc = network.accuracy(x_train, t_train)#计算精度
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))














