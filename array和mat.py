Numpy包中两种常见数值运算类型array和mat比较

Numpy包广泛用于python的数值计算中，因此有必要对其进行熟悉。

首先记住，shape array mat都是numpy模块中的方法，而不是变量的方法，因此调用方式都是np.array() np.mat() np.shape()，没有a.shape()的说法，正确姿势应该是m,n=np.shape(a),如果a有二维的话。
下面说明numpy中mat array类型的区别。
a=[[1,2,3],[4,5,6]]
a=np.array(a)

a                               array，将列表数组化，数组的每个元素是原列表的元素
array([[1, 2, 3],
       [4, 5, 6]])
type(a)
<class 'numpy.ndarray'>

a.transpose()               两者都有transpose方法
array([[1, 4],
       [2, 5],
       [3, 6]])

b=np.array([1,2,2])

b
array([1, 2, 2])

a*b
array([[ 1,  4,  6],
       [ 4, 10, 12]])            并非按照矩阵乘法的规则计算，而是对应元素直接相乘了


c=np.mat([[1,2,3],[4,5,6]])    mat，将列表矩阵化
c
matrix([[1, 2, 3],
        [4, 5, 6]])

d=np.mat([1,2,2])
d
matrix([[1, 2, 2]])                 注意b和d的区别，d比b多了一层中括号

lenb=np.shape(b)
lenb
(3,)                                    shape方法返回了一个元组，元组只有一个元素

lenb=np.shape(b)[0]
lenb
3                                        第一个也是唯一一个元素，为3

lenb=np.shape(b)[1]                                    没有第二个元素，故报错。
Traceback (most recent call last):
  File "D:\Program Files (x86)\JetBrains\PyCharm Edu 2017.3\helpers\pydev\_pydevd_bundle\pydevd_exec2.py", line 3, in Exec
    exec(exp, global_vars, local_vars)
  File "<input>", line 1, in <module>
IndexError: tuple index out of range

d
matrix([[1, 2, 2]])

lend=np.shape(d)                    d是矩阵，故有两个维度；而b是数组，shape相当于返回了b的长度
lend
(1, 3)
但是 
lena=np.shape(a)
lena
(2, 3)                                         数组中元素个数大于1时，array的表现和mat一样
而且
c*d
Traceback (most recent call last):
  File "D:\Program Files (x86)\JetBrains\PyCharm Edu 2017.3\helpers\pydev\_pydevd_bundle\pydevd_exec2.py", line 3, in Exec
    exec(exp, global_vars, local_vars)
  File "<input>", line 1, in <module>
  File "D:\Anaconda2\envs\py3\lib\site-packages\numpy\matrixlib\defmatrix.py", line 309, in __mul__
    return N.dot(self, asmatrix(other))
ValueError: shapes (2,3) and (1,3) not aligned: 3 (dim 1) != 1 (dim 0)

由于c和d都是矩阵，不再满足数组一样的对应相乘，而是遵循矩阵乘法原则：

c*d.transpose()
matrix([[11],
        [26]])

数组a和矩阵c都可以使用[i,j]的方式引用元素，其索引方式与c语言相通（从0开始）
--------------------- 
