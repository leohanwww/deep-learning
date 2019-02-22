数组索引

当被索引的数组 a 是一个多维数组，单个索引数组指的是 a 的第一个维度。
怎么理解呢

>>> a = np.arange(12).reshape(3,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> i = np.array( [ [0,1],                        # indices for the first dim of a
...                 [1,2] ] )
>>> j = np.array( [ [2,1],                        # indices for the second dim
...                 [3,3] ] 


>>> a[i]
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],

       [[ 4,  5,  6,  7],
        [ 8,  9, 10, 11]]])
>>> a[i,j]                                     # i and j must have equal shape
array([[ 2,  5],
       [ 7, 11]])
i和j中所有的数字都代表a中的第一个维度，这里是行，i和j组成索引矩阵[0,2][1,1][1,3][2,3]

>>> s = np.array( [i,j] )
>>> s
array([[[0, 1],
        [1, 2]],

       [[2, 1],
        [3, 3]]])
>>> a[s]                                       # not what we want
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
IndexError: index (3) out of range (0<=index<=2) in dimension 0
这样不行因为取a的索引会认为是第一维度的
>>> a[tuple(s)]                                # same as a[i,j]
array([[ 2,  5],
       [ 7, 11]])

>>> time = np.linspace(20, 145, 5)                 # time scale
>>> data = np.sin(np.arange(20)).reshape(5,4)      # 4 time-dependent series
>>> time
array([  20.  ,   51.25,   82.5 ,  113.75,  145.  ])
>>> data
array([[ 0.        ,  0.84147098,  0.90929743,  0.14112001],
       [-0.7568025 , -0.95892427, -0.2794155 ,  0.6569866 ],
       [ 0.98935825,  0.41211849, -0.54402111, -0.99999021],
       [-0.53657292,  0.42016704,  0.99060736,  0.65028784],
       [-0.28790332, -0.96139749, -0.75098725,  0.14987721]])


你还可以使用数组索引作为目标来赋值：

>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> a[[1,3,4]] = 0
>>> a
array([0, 0, 2, 0, 0])

>>> a = np.arange(12).reshape(3,4)
>>> b = a > 4
>>> b                                          # b is a boolean with a's shape
array([[False, False, False, False],
       [False,  True,  True,  True],
       [ True,  True,  True,  True]])
>>> a[b]                                       # 1d array with the selected elements
array([ 5,  6,  7,  8,  9, 10, 11])

>>> a[b] = 0                                   # All elements of 'a' higher than 4 become 0
>>> a
array([[0, 1, 2, 3],
      [4, 0, 0, 0],
      [0, 0, 0, 0]])

>>> a = np.arange(12).reshape(3,4)
>>> b1 = np.array([False,True,True])             # first dim selection
>>> b2 = np.array([True,False,True,False])       # second dim selection
>>>
>>> a[b1,:]                                   # selecting rows
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> a[b1]                                     # same thing
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> a[:,b2]                                   # selecting columns
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])
>>>
>>> a[b1,b2]                                  # a weird thing to do
array([ 4, 10])


>>> u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
>>> u
array([[ 1.,  0.],
       [ 0.,  1.]])
>>> np.trace(u)  # trace 计算斜方向的和
2.0

a = np.array([[1.0,2.0],[3.0,4.0]])
>>> np.linalg.inv(a) # 计算逆矩阵
array([[-2. ,  1. ],
       [ 1.5, -0.5]])

y = np.array([[5.], [7.]])
>>> np.linalg.solve(a, y) #求解多元一次方程
array([[-3.],
       [ 4.]])

基本切片
基本切片语法是 i:j:k 其中i是起始索引，j是停止索引，k是步骤（k\neq0）
>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> x[1:7:2]
array([1, 3, 5])

>>> x[-2:10]
array([8, 9])
>>> x[-3:3:-1]
array([7, 6, 5, 4])

>>> x[5:]
array([5, 6, 7, 8, 9])

>>> x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
>>> x.shape
(2, 3, 1)
>>> x[1:2]
array([[[4],
        [5],
        [6]]])
>>> x.shape
(2, 3, 1)
>>> x[...,0] #x.ndim生成与长度相同的选择元组所需的对象数。和x[:,:,0]效果一样
array([[1, 2, 3],
    [4, 5, 6]])
>>> x.shape
(2, 3)
>>> x[:,np.newaxis,:,:].shape #增加维度
(2, 1, 3, 1)

高级索引

>>> x = np.array([[1, 2], [3, 4], [5, 6]])
>>> x[[0, 1, 2], [0, 1, 0]]
array([1, 4, 5])

>>> x = array([[ 0,  1,  2],
...            [ 3,  4,  5],
...            [ 6,  7,  8],
...            [ 9, 10, 11]])
>>> rows = np.array([[0, 0],
...                  [3, 3]], dtype=np.intp)
>>> columns = np.array([[0, 2],
...                     [0, 2]], dtype=np.intp)
>>> x[rows, columns] #此方法等同x[:2,:2]，意在取x的指定子数组
array([[ 0,  2],
       [ 9, 11]])

>>> rows = np.array([0, 3], dtype=np.intp)
>>> columns = np.array([0, 2], dtype=np.intp)
>>> rows[:, np.newaxis]
array([[0],
       [3]])
>>> x[rows[:, np.newaxis], columns]
array([[ 0,  2],
       [ 9, 11]])
使用函数ix_也可以实现这种广播
>>> x[np.ix_(rows, columns)]
array([[ 0,  2],
       [ 9, 11]])
#如果没有np.ix_进行广播，就只取[0,0]和[3,2]两个元素


布尔数组索引
>>> x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
x[np.isnan(x)]
array([ nan,  nan,  nan])
>>> x[~np.isnan(x)]
array([ 1.,  2.,  3.])
>>> x = np.array([1., -1., -2., 3])
>>> x[x<0] = 99
>>> x
array([  1.,  99.,  99.,   3.])

>>> x = np.array([[0, 1], [1, 1], [2, 2]])
>>> rowsum = x.sum(-1) #按行计算和，这里我用x.sum(1)也一样
>>> x[rowsum <= 2, :]
array([[0, 1],
       [1, 1]])

>>> x = array([[ 0,  1,  2],
...            [ 3,  4,  5],
...            [ 6,  7,  8],
...            [ 9, 10, 11]])
>>> rows = (x.sum(-1) % 2) == 0
>>> rows
array([False,  True, False,  True])
>>> columns = [0, 2]
>>> x[np.ix_(rows, columns)]
array([[ 3,  5],
       [ 9, 11]])

x.flat返回一个迭代器，它将迭代整个数组，x.flat是一维视图，x.flat[:]调用









































