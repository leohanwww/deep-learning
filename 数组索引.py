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




































































































