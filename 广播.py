广播


当在两个数组上操作时，NumPy在元素级别比较它们的形状。它从尾随的维度开始，并朝着前进的方向前进。两个维度兼容，当

1.他们是平等的，或者
2.其中之一是1
如果不满足这些条件，则抛出ValueError: frames are not aligned异常，指示数组具有不兼容的形状。
结果数组的大小是沿着输入数组的每个维度的最大大小。

A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
B的大小为1的维度被拉伸以适应A，A的大小为1的维也被拉伸以适应
>>> x = np.arange(4)
>>> xx = x.reshape(4,1)
>>> y = np.ones(5)
>>> z = np.ones((3,4))

>>> x + y
<type 'exceptions.ValueError'>: shape mismatch: objects cannot be broadcast to a single shape
>>> xx + y
array([[ 1.,  1.,  1.,  1.,  1.],
       [ 2.,  2.,  2.,  2.,  2.],
       [ 3.,  3.,  3.,  3.,  3.],
       [ 4.,  4.,  4.,  4.,  4.]])
>>> x + z
array([[ 1.,  2.,  3.,  4.],
       [ 1.,  2.,  3.,  4.],
       [ 1.,  2.,  3.,  4.]])


A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4

A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5











































































































