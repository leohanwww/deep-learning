IO操作

import numpy as np
from io import BytesIO

>>> data = "1, 2, 3\n4, 5, 6"
>>> np.genfromtxt(BytesIO(data), delimiter=',')
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])

>>> data = "  1  2  3\n  4  5 67\n890123  4"
>>> np.genfromtxt(BytesIO(data), delimiter=3) #定界符为3 意思是遇到3就分隔，换行是/n决定的
array([[   1.,    2.,    3.],
       [   4.,    5.,   67.],
       [ 890.,  123.,    4.]])
>>> data = "123456789\n   4  7 9\n   4567 9"
>>> np.genfromtxt(BytesIO(data), delimiter=(4, 3, 2))
array([[ 1234.,   567.,    89.],
       [    4.,     7.,     9.],
       [    4.,   567.,     9.]])



>>> data = "1, abc , 2\n 3, xxx, 4"
>>> # Without autostrip
>>> np.genfromtxt(BytesIO(data), delimiter=",", dtype="|S5")
array([['1', ' abc ', ' 2'],
       ['3', ' xxx', ' 4']],
      dtype='|S5')
>>> # With autostrip
>>> np.genfromtxt(BytesIO(data), delimiter=",", dtype="|S5", autostrip=True)
array([['1', 'abc', '2'],
       ['3', 'xxx', '4']],
      dtype='|S5')
#autostrip取消单个元素前后的空白


>>> data = """#
... # Skip me !
... # Skip me too !
... 1, 2
... 3, 4
... 5, 6 #This is the third line of the data
... 7, 8
... # And here comes the last line
... 9, 0
... """
>>> np.genfromtxt(BytesIO(data), comments="#", delimiter=",")
[[ 1.  2.]
 [ 3.  4.]
 [ 5.  6.]
 [ 7.  8.]
 [ 9.  0.]]

>>> data = "\n".join(str(i) for i in range(10))
>>> np.genfromtxt(BytesIO(data),)
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
>>> np.genfromtxt(BytesIO(data),
...               skip_header=3, skip_footer=5)
array([ 3.,  4.])

>>> data = "1 2 3\n4 5 6"
>>> np.genfromtxt(BytesIO(data), usecols=(0, -1))
array([[ 1.,  3.],
       [ 4.,  6.]])
>>> data = "1 2 3\n4 5 6"
>>> np.genfromtxt(BytesIO(data),
...               names="a, b, c", usecols=("a", "c"))


数据的类型dtype
单一 dtype=float
系列类型 dtype=(int, float, float)
逗号分隔的额类型 dtype="i4,f8,|S3"
包含字典
>>> np.genfromtxt(BytesIO(data),
...               names="a, b, c", dtype=[('a','int'),('b','float')])

names参数
分配每一列的名称
>>> data = BytesIO(b'1 2 3\n 4 5 6')
>>> np.genfromtxt(data, name="A, B, C")
array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
      dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> ndtype=[('a',int), ('b', float), ('c', int)]
>>> names = ["A", "B", "C"]
>>> np.genfromtxt(data, names=names, dtype=ndtype)
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('A', '<i8'), ('B', '<f8'), ('C', '<i8')])

如果 names=None 的时候，只是预计会有一个结构化的dtype，它的名称将使用标准的NumPy默认值 "f%i"来定义，会产生例如f0，f1等名称：
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> np.genfromtxt(data, dtype=(int, float, int))
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', '<i8')])

我们可以使用defaultfmt参数覆盖此默认值，该参数采用任何格式字符串：
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> np.genfromtxt(data, dtype=(int, float, int), defaultfmt="var_%02i")
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('var_00', '<i8'), ('var_01', '<f8'), ('var_02', '<i8')])

converters参数

>>> convertfunc = lambda x: float(x.strip("%"))/100.
>>> data = "1, 2.3%, 45.\n6, 78.9%, 0"
>>> names = ("i", "p", "n")
>>> np.genfromtxt(BytesIO(data), delimiter="," names=names, converters={1:convertfunc})
array([(1.0, 0.023, 45.0), (6.0, 0.78900000000000003, 0.0)],
      dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])
通过使用第二列（"p"）作为关键字而不是其索引（1）的名称，可以获得相同的结果
>>> # Using a name for the converter ...
>>> np.genfromtxt(BytesIO(data), delimiter=",", names=names,
...               converters={"p": convertfunc})
array([(1.0, 0.023, 45.0), (6.0, 0.78900000000000003, 0.0)],
      dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])

missing_values和filling_values

>>> data = "N/A, 2, 3\n4, ,???"
>>> kwargs = dict(delimiter=",",
dtype=int
names="a,b,c"
missing_values={0:"N/A", 'b':" ", '2':'???'}
filling_values={0:0, 'b':0, 2:-999})
>>> np.genfromtxt(BytesIO(data), **kwargs)
array([(0, 2, 3), (4, 0, -999)],
      dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<i8')])


























