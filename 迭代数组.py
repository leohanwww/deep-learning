迭代数组

单数组迭代
a = np.arange(6).reshape(2,3)
for x in np.nditer(a):# 访问数组的每个元素
	print x
不同的迭代顺序
for x in np.nditer(a, order='F'):
	print x
0 3 1 4 2 5
>>> for x in np.nditer(a.T, order='C'):
...     print x,
...
0 3 1 4 2 5

修改数值
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> for x in np.nditer(a, op_flags=['readwrite']):
...     x[...] = 2 * x 
#Python中的常规赋值只是更改本地或全局变量字典中的引用，而不是修改现有变量。 这意味着简单地分配给x不会将值放入数组的元素中，而是将x作为数组元素引用切换为对指定值的引用。 要实际修改数组的元素，x应该用省略号索引。
>>> a
array([[ 0,  2,  4],
       [ 6,  8, 10]])

外部循环

>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a, flags=['external_loop']):
...     print(x)
...
[0 1 2 3 4 5]
>>> for x in np.nditer(a, flags=['external_loop'], order='F'):
...     print(x),
...
[0 3] [1 4] [2 5]

索引
>>> a = np.arange(6).reshape(2,3)
>>> it = np.nditer(a, flags=['f_index'])#设置此flags才能有index属性
>>> while not it.finished:
...     print "%d < %d>" % (it[0], it.index),
...     it.iternext()
...
0 <0> 1 <2> 2 <4> 3 <1> 4 <3> 5 <5>

>>> it = np.nditer(a, flags=['multi_index'])
>>> while not it.finished:
...     print "%d < %s>" % (it[0], it.multi_index),
...     it.iternext()
...
0 <(0, 0)> 1 <(0, 1)> 2 <(0, 2)> 3 <(1, 0)> 4 <(1, 1)> 5 <(1, 2)>

>>> it = np.nditer(a, flags=['multi_index'], op_flags=['writeonly'])
>>> while not it.finished:
...     it[0] = it.multi_index[1] - it.multi_index[0]
...     it.iternext()
...
>>> a
array([[ 0,  1,  2],
       [-1,  0,  1]])

跟踪索引或多索引与使用外部循环不兼容，不能同时使用
>>> a = np.zeros((2,3))
>>> it = np.nditer(a, flags=['c_index', 'external_loop'])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked

缓冲数组元素
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a, flags=['external_loop'], order='F'):
...     print x,
...#不做缓冲会降低性能
[0 3] [1 4] [2 5]
>>> for x in np.nditer(a, flags=['external_loop','buffered'], order='F'):
...     print x,
...
[0 3 1 4 2 5]

迭代为特定数据类型
>>> a = np.arange(6).reshape(2,3) - 3
>>> for x in np.nditer(a, op_flags=['readonly','copy'],
...                 op_dtypes=['complex128']):
...     print np.sqrt(x)
...
1.73205080757j 1.41421356237j 1j 0j (1+0j) (1.41421356237+0j)

>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['complex128']):
...     print np.sqrt(x),
...
1.73205080757j 1.41421356237j 1j 0j (1+0j) (1.41421356237+0j)


#执行安全转换same_kind，只允许可以转换的数据类型
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['float32'],
...                 casting='same_kind'):
...     print x,
...
0.0 1.0 2.0 3.0 4.0 5.0
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['int32'], casting='same_kind'):
...     print x,
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Iterator operand 0 dtype could not be cast from dtype('float64') to dtype('int32') according to the rule 'same_kind'

广播数组迭代

>>> a = np.arange(3)
>>> b = np.arange(6).reshape(2,3)
>>> for x, y in np.nditer([a,b]):
...     print "%d:%d" % (x,y),
...
0:0 1:1 2:2 0:3 1:4 2:5

>>> def square(a, out=None):
...     it = np.nditer([a, out],
...             flags = ['external_loop', 'buffered'],
...             op_flags = [['readonly'],
...                         ['writeonly', 'allocate', 'no_broadcast']])
...     for x, y in it:
...         y[...] = x*x
...     return it.operands[1]
...
>>> square([1,2,3])
array([1, 4, 9])
>>> b = np.zeros((3,))
>>> square([1,2,3], out=b)
array([ 1.,  4.,  9.])
>>> b
array([ 1.,  4.,  9.])


外部产品迭代

>>> a = np.arange(3)
>>> b = np.arange(8).reshape(2,4)
>>> it = np.nditer([a, b, None], flags=['external_loop'],
...             op_axes=[[0, -1, -1], [-1, 0, 1], None])
>>> for x, y, z in it:
...     z[...] = x*y
...
>>> it.operands[2]
array([[[ 0,  0,  0,  0],
        [ 0,  0,  0,  0]], # a中的 0 * b全部
       [[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]], # a中的 1 * b全部
       [[ 0,  2,  4,  6],
        [ 8, 10, 12, 14]]]) # a中的 2 * b全部

减少迭代

>>> a = np.arange(24).reshape(2,3,4)
>>> b = np.array(0)
>>> for x, y in np.nditer([a, b], flags=['reduce_ok', 'external_loop'],
...                     op_flags=[['readonly'], ['readwrite']]):
...     y[...] += x
...
>>> b
array(276)
>>> np.sum(a)
276



































































































































































































