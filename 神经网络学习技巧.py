神经网络学习技巧

参数的更新
寻找使得损失达到尽可能小的参数
，为了找到最优参数，我们将参数的梯度（导数）作为了线索。
使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠
近最优参数，这个过程称为随机梯度下降法（stochastic gradient descent），
简称SGD

class SGD:
	def __init__(self,lr=0.01)
		self.lr = lr

	def upgrade(self,params,grades)
		for key in params.keys():
			params[key] -= self.lr * grades[key]
			
for i in range(10000):
	network = TwoLayerNet()
	optimizer = SGD()
	params = network.params
	optimizer.upgrade(params,grades)

Momentum（动量）

class Momentum:
	
	def __init__(self, lr=0.01, momentum=0.9):
		self.lr = lr
		self.momentum = momentum
		self.v = None

	def upgrade(self, params, grades):
		if self.v is None
			self.v = {}
			for key, val in params.items():
				self.v[key] = np.zeros_like(val)

		for key in params.keys():
			self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
			params[key] += self.v[key]
#实例变量v会保存物体的速度。初始化时，v中什么都不保存，但当第一次调用update()时，v会以字典型变量的形式保存与参数结构相同的数据。

AdaGrad

class AdaGrad:
	def __init__(self, lr=0.01):
		self.lr = lr
		self.h = None

	def update(self, params, grads):
		if self.h is None:
			self.h = {}
		for key, val in params.items():
			self.h[key] = np.zeros_like(val)

	for key in params.keys():
		self.h[key] += grads[key] * grads[key]
		params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

Adam
....
四种更新参数的方式：Momentum、AdaGrad、Adam比SGD学习进行的速度要快些


权重的初始值
初始的权重不能为0，
为什么不能将权重初始值设为0 呢？严格地说，为什么不能将权重初始
值设成一样的值呢？这是因为在误差反向传播法中，所有的权重值都会进行
相同的更新。比如，在2 层神经网络中，假设第1 层和第2 层的权重为0。这
样一来，正向传播时，因为输入层的权重为0，所以第2 层的神经元全部会
被传递相同的值。第2 层的神经元中全部输入相同的值，这意味着反向传播
时第2 层的权重全部都会进行相同的更新。因此，权重被更新为相同的值，并拥有了对称的值（重复的值）。
这使得神经网络拥有许多不同的权重的意义丧失了。为了防止“权重均一化”
（严格地讲，是为了瓦解权重的对称结构），必须随机生成初始值。

权重的规划要使得每个神经元输出不同的值，因为如果100个神经元输出几乎相同的值，也能用1个神经元表示相同的值

xavier 参数：1/np.sqrt(n)
node_num = 100 # 前一层的节点数
w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

ReLu层的‘He初始值’：当前一层的节点数为n 时，He 初始值使用标准差为np.sqrt(2/n)的高斯分布

Batch Normalization 数据正规化，使数据分布的均值为0、方差为1


防止过拟合的方法：过拟合指的是只能拟
合训练数据，但不能很好地拟合不包含在训练数据中的其他数据的状态

超参数的测定

weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)






















































































































