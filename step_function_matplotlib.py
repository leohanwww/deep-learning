import numpy as np
import matplotlib.pylab as plt

def step_function(x):
	y = x > 0 # x  np.array([-1,1,2]) y  array(False,True,True)
	return y.astype(np.int) #y array(0,1,1)
	
	
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
	
x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #调整y轴的范围
plt.show()