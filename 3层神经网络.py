3层神经网络

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
	
def init_network():
	network = {}
	network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
	network['b1'] = np.array([0.1, 0.2, 0.3])
	network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
	network['b2'] = np.array([0.1, 0.2])
	network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
	network['b3'] = np.array([0.1, 0.2])
	return network

def forward(X, network):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']
	a1 = np.dot(X, W1)
	z1 = sigmod(a1)
	a2 = np.dot(z1, W2)
	z2 = sigmod(a2)
	a3 = np.dot(z2, W3)
	y = identity_function(a3)
	return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(x, network)
print(y)
[ 0.31682708 0.69627909]


输出层的设计
输出层所用的激活函数，要根据求解问题的性质决定。一般地，回
归问题可以使用恒等函数，二元分类问题可以使用sigmoid 函数，
多元分类问题可以使用softmax 函数。
机器学习的问题大致可以分为分类问题和回归问题。分类问题是数
据属于哪一个类别的问题。比如，区分图像中的人是男性还是女性
的问题就是分类问题。而回归问题是根据某个输入预测一个（连续的）
数值的问题。比如，根据一个人的图像预测这个人的体重的问题就
是回归问题（类似“57.4kg”这样的预测）。

机器学习的步骤可以分为学习和推理两个阶段
1 学习阶段：这里的“学习”是指使用训练数据、自动调整参数的过程
2.推理阶段：用学到的模型对未知的数据进行推理（分类）。


手写数字识别
学习完成进行推断
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()
	
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
normalize=False) #x_train.shape (60000,784) t_train.shape (60000,)
img = x_train[0] #选取第一张图片
label = t_train[0]#选取第一个监督数据

print(label) # 5 这里的正确数据应该是数字5
print(img.shape) # (784,)
img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
print(img.shape) # (28, 28)

img_show(img)

开始推测
def get_data():
	(x_train, t_train), (x_test, t_test) = \
		load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test

def init_network():#已经学习到的权重参数
	with open("sample_weight.pkl", 'rb') as f:
		network = pickle.load(f)
	return network

def predict(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = softmax(a3)
	return y #y.shape (10,)

x,t = get_data()
network = init_network()

accuracy_cnt = 0 #初始化准确次数
for i in range(len(x)): #x.shape (10000,784)
	y = predict(network, x[i])
	p = np.argmax(y) #选择概率最高的元素的索引
	if p == t[i]#判断结果正确
		accuracy_cnt += 1

print(accuracy_cnt / len(x))#准确率

批处理
>>> x, _ = get_data()
>>> network = init_network()
>>> W1, W2, W3 = network['W1'], network['W2'], network['W3']
>>> x.shape
(10000, 784)
>>> x[0].shape
(784,)
>>> W1.shape
(784, 50)
>>> W2.shape
(50, 100)
>>> W3.shape
(100, 10)

x, t = get_data()
network = init_network()
batch_size = 100 # 批数量
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
	x_batch = x[i, i+batch_size] # x[0:100] x[100:200]...
	y_batch = predict(network, x_batch) #y.shape (100,10)
	p = np.argmax(y_batch, axis=1) #第一维即行方向
	accuracy_cnt += np.sum(p == t[i,i+batch_size])

矩阵的第0 维是列方向，第1 维是行方向


