卷积神经网络

卷积层进行卷积运算，卷积运算对输入数据应用滤波器

在4X4的输入数据里卷积3X3的滤波器，卷积单体运算是4x4内的3x3和滤波器的3x3按行相乘计算所有和

填充
在大小的4x4的周围使用幅度为1的填充（在外围填充0）使得变成6x6的输入，然后应用大小为3x3的滤波器，生成大小为4x4的输出
使用填充主要是为了调整输出的大小。比如，对大小为(4, 4) 的输入
数据应用(3, 3) 的滤波器时，输出大小变为(2, 2)，相当于输出大小
比输入大小缩小了2 个元素。这在反复进行多次卷积运算的深度网
络中会成为问题。为什么呢？因为如果每次进行卷积运算都会缩小
空间，那么在某个时刻输出大小就有可能变为1，导致无法再应用
卷积运算。为了避免出现这样的情况，就要使用填充。在刚才的例
子中，将填充的幅度设为1，那么相对于输入大小(4, 4)，输出大小
也保持为原来的(4, 4)。因此，卷积运算就可以在保持空间大小不变
的情况下将数据传给下一层。

步幅
应用滤波器的位置间隔称为步幅
增大步幅后，输出大小会变小。而增大填充后，输出大小会变大

卷积层输出大小计算
假设输入大小为(H,W)，滤波器大小为(FH, FW)，输出大小为
(OH,OW)，填充为P，步幅为S，输出公式见：
OH = ((H + 2P -FH) / S) + 1
OW = ((W + 2P -FW) / S) + 1

池化层输出大小计算,F是池化层的核大小，S是池化层的步幅
PW = (W - F)/S + 1
PH = (H - F)/s + 1

3维数据的卷积运算
通道是第三维度的长度，滤波器的通道（第三维）要和输入数据的一样

将数据和滤波器结合长方体的方块来考虑,C是通道数，H和W为高和宽，（C，H，W）卷积（C，FH，FW）得到（1，OH，OW）的输出
如果要在通道方向上也有多个卷积输出，就需要多个滤波器（FN，C，FH，FW）一个输入卷积多个滤波器，加上（FN，1，1）的偏置得到（FN，OH，OW）的输出，这就是CNN的处理流

批处理
输入数据（N，C，H，W）卷积 滤波器（FN，C，FH，FW）		N个数据（N，FN，OH，OW）	偏置（FN，1，1）		得到（N，FN，OH，OW）


池化层
缩小高和长方向上的空间，2x2的区域中取出最大的元素（MAX池化），此时步幅为2，池化的窗口大小一般和步幅相同

卷积层的实现
4维数组
>>>x = np.random.randon(10,1,28,28)
>>>x.shape
(10,1,28,28)
>>> x[0].shape # (1, 28, 28)
>>> x[1].shape # (1, 28, 28)

基于im2col 的展开
4维数据看上去很复杂，而且多循环不好，im2col是一个函数“image to column”，将输入数据展开以适合滤波器（权重）
im2col按照步幅，将滤波器的应用区域从头开始依次横向展开为1 列
使用im2col展开输入数据后，之后就只需将卷积层的滤波器（权重）纵
向展开为1 列，并计算2 个矩阵的乘积即可


import sys, os
sys.path.append(os.pardir)
from common.util import im2col
x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9, 75)
x2 = np.random.rand(10, 3, 7, 7) # 10个数据
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90, 75)
这里举了两个例子。第一个是批大小为1、通道为3 的7 × 7 的数据，第
二个的批大小为10，数据形状和第一个相同。分别对其应用im2col函数，在
这两种情形下，第2 维的元素个数均为75。这是滤波器（通道为3、大小为
5 × 5）的元素个数的总和。批大小为1 时，im2col的结果是(9, 75)。而第2
个例子中批大小为10，所以保存了10 倍的数据，即(90, 75)。

class Convolution:
	def __init__(self, W, b, stride=1, pad=0):
		self.W = W
		self.b = b
		self.stride = stride
		self.pad = pad
#卷积层的初始化方法将滤波器（权重）、偏置、步幅、填充作为参数接收。
#滤波器是(FN, C, FH, FW) 的4 维形状。
#另外，FN、C、FH、FW 分别是Filter Number（滤波器数量）、Channel、Filter Height、Filter Width 的缩写。

	def forward(self, x):
		FN, C, FH, FW = self.W.shape#滤波器的维度
		N, C, H, W = x.shape#输入数据的维度
		out_h = int(1 + (H + 2*self.pad - FH) / self.stride)#计算输出的高度
		out_w = int(1 + (W + 2*self.pad - FW) / self.stride)#输出的宽度

		col = im2col(x, FH, FW, self.stride, self.pad)#展开数据
		col_W = self.W.reshape(FN, -1).T # 滤波器展开为FN行再转置
		out = np.dot(col, col_W) + self.b

		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
		#（N,H,W,C）	(N,C,H,W)
		return out

池化层的实现
池化的应用区域按通道单独展开		2x2的方块变成1x4的一行，再从每行取max	reshape成（C，H，W）形状

class Polling:
	def __init__(self, pool_h, pool_w, stride=1, pad=0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad
		
	def forward(self, x):
		N, C, H, W = x.shape
		out_h = int(1 + (H - self.pool_h) / self.stride)
		out_w = int(1 + (W - self.pool_w) / self.stride)

		# 展开(1)
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)
		# 最大值(2)
		out = np.max(col, axis=1)
		# 转换(3)
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
		return out


CNN的实现
• input_dim―输入数据的维度：（通道，高，长）
• conv_param―卷积层的超参数（字典）。字典的关键字如下：
filter_num―滤波器的数量
filter_size―滤波器的大小
stride―步幅
pad―填充
• hidden_size―隐藏层（全连接）的神经元数量
• output_size―输出层（全连接）的神经元数量
• weitght_int_std―初始化时权重的标准差
class SimpleConvNet:
	
	def __init__(self, input_dim=(1,28,28), 
			conv_parm={
			'filter_num':30,'filter_size':5,'strde':1,'pad':0}
			hidden_size=100, output_size=10, weight_init_std=0.01):
			
		
		#过滤层初始化
		filter_num = conv_param['filter_num']
		filter_size = conv_param['filter_size']
		filter_pad = conv_param['pad']
		filter_stride = conv_param['stride']
		input_size = input_dim[1]
		conv_output_size = (input_size - filter_size + 2*filter_pad) / \
		filter_stride + 1#卷积层的输出大小
		pool_output_size = int(filter_num * (conv_output_size/2) *
		(conv_output_size/2))#池化层的输出大小
		#权重初始化
		self.params = {}
		self.params['W1'] = weight_init_std * \
		np.random.randn(filter_num, input_dim[0],
		filter_size, filter_size)
		self.params['b1'] = np.zeros(filter_num)
		self.params['W2'] = weight_init_std * \
		np.random.randn(pool_output_size,
		hidden_size)
		self.params['b2'] = np.zeros(hidden_size)
		self.params['W3'] = weight_init_std * \
		np.random.randn(hidden_size, output_size)
		self.params['b3'] = np.zeros(output_size)

		self.layers = OrderedDict()
		self.layers = Convolution(self.params['W1'],
									self.params['b1'],
									self.params['stride']
									self.params['pad']
		self.layers['Relu1'] = Relu()
		self.layers['Pool1'] = Polling(pool_h=2,pool_w=2,stride=2)
		self.layers['Affine1'] = Affine(self.params['W3'],self.params['b3'])
		self.layers['Relu2'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W3'],self.params['b3'])
		self.last_layer = softmaxwithloss()

	def predict(self,x):
		for layer in self.laysers.values():
			layer.forward(x)
		return x

	def loss(self,x,t):
		y = self.predict(x)
		return self.last_layer.forward(y,t)

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
		grads['W1'] = self.layers['Conv1'].dW
		grads['b1'] = self.layers['Conv1'].db
		grads['W2'] = self.layers['Affine1'].dW
		grads['b2'] = self.layers['Affine1'].db
		grads['W3'] = self.layers['Affine2'].dW
		grads['b3'] = self.layers['Affine2'].db
		return grads





























































































































































































































