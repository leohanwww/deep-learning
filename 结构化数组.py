结构化数组
结构化数组其实就是ndarrays

>>> x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
...              dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
>>> x
array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
      dtype=[('name', 'S10'), ('age', '<i4'), ('weight', '<f4')])

可以通过使用字段名称进行索引来访问和修改结构化数组的各个字段的值
>>> x['age']
array([9, 3], dtype=int32)
>>> x['age'] = 5
>>> x
array([('Rex', 5, 81.0), ('Fido', 5, 27.0)],
      dtype=[('name', 'S10'), ('age', '<i4'), ('weight', '<f4')])
>>> x[1]
('Fido', 3, 27.0)
>>> x[1]['age']
3
>>> x['age'] = 5
>>> x
array([('Rex', 5, 81.0), ('Fido', 5, 27.0)],
      dtype=[('name', 'S10'), ('age', '<i4'), ('weight', '<f4')])


结构化数据类型可以被认为是一定长度的字节序列（结构的itemsize），它被解释为一个字段集合。 每个字段在结构中都有一个名称，一个数据类型和一个字节偏移量。 字段的数据类型可以是任何numpy数据类型，包括其他结构化数据类型，它也可以是一个子数组，其行为类似于指定形状的ndarray。 字段的偏移是任意的，并且字段甚至可以重叠。 这些偏移通常由numpy自动确定，但也可以指定。
结构化数据类型可以使用函数numpy.dtype创建
元组属性：每个元组都有这些属性（fieldname，datatype，shape）
如果fieldname是空字符串''，那么该字段将被赋予一个默认名称形式f#
>>> np.dtype([('x', 'f4'),('', 'i4'),('z', 'i8')])
 dtype([('x', '<f4'), ('f1', '<i4'), ('z', '<i8')])

>>> np.dtype('i8,f4,S3')
 dtype([('f0', '<i8'), ('f1', '<f4'), ('f2', 'S3')])
#不写名称默认(f0,f1...)创建

字典参数
>>> np.dtype({'names': ['col1', 'col2'], 'formats': ['i4','f4']})
 dtype([('col1', '<i4'), ('col2', '<f4')])

>>> np.dtype({'names': ['col1', 'col2'],
 ...           'formats': ['i4','f4'],
 ...           'offsets': [0, 4],
 ...           'itemsize': 12})
 dtype({'names':['col1','col2'], 'formats':['<i4','<f4'], 'offsets':[0,4], 'itemsize':12})

>>> d = np.dtype([('x', 'i8'), ('y', 'f4')])
>>> d.names
('x', 'y')

字段标题
要在使用dtype规范的list-of-tuples形式时添加标题，可以将字段名称指定为两个字符串的元组而不是单个字符串，它们分别是字段的标题和字段名称
>>> np.dtype([(('my title', 'name'), 'f4')])


将数据分配给数组
>>> x = np.array([(1,2,3),(4,5,6)], dtype='i8,f4,f8')
>>> x[1] = (7,8,9)
>>> x
array([(1, 2., 3.), (7, 8., 9.)],
     dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '<f8')])
通过标量赋值
>>> x = np.zeros(2, dtype='i8,f4,?,S1')
>>> x[:] = 3
>>> x
array([(3, 3.0, True, b'3'), (3, 3.0, True, b'3')],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])
>>> x[:] = np.arange(2)
>>> x
array([(0, 0.0, False, b'0'), (1, 1.0, True, b'1')],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])

结构化数组也可以分配给非结构化数组，但前提是结构化数据类型只有一个字段
>>> twofield = np.zeros(2, dtype=[('A', 'i4'), ('B', 'i4')])
>>> onefield = np.zeros(2, dtype=[('A', 'i4')])
>>> nostruct = np.zeros(2, dtype='i4')
>>> nostruct[:] = twofield
ValueError: Can't cast from structure to non-structure, except if the structure only has a single field.
#不能这样赋值，因为twofield有两个字段类型
>>> nostruct[:] = onefield
>>> nostruct
array([0, 0], dtype=int32)

两个结构化数组之间的分配就像源元素已转换为元组然后分配给目标元素一样。 也就是说，源阵列的第一个字段分配给目标数组的第一个字段，第二个字段同样分配，依此类推，而不管字段名称如何。 具有不同数量的字段的结构化数组不能彼此分配。 未包含在任何字段中的目标结构的字节不受影响。
>>> a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])
>>> b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])
>>> b[:] = a
>>> b
array([(0.0, b'0.0', b''), (0.0, b'0.0', b''), (0.0, b'0.0', b'')],
      dtype=[('x', '<f4'), ('y', 'S3'), ('z', 'O')])#赋值后的目标字段类型不变，只拷贝源的数据



索引结构化数组

字段名访问
>>> x = np.array([(1,2),(3,4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
>>> x['foo']
array([1, 3])
>>> x['foo'] = 10
>>> x
array([(10, 2.), (10, 4.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])

>>> y[:] = 10
>>> y
array([ 10.,  10.], dtype=float32)
>>> x
array([(1, 10.0), (3, 10.0)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])

多字段访问
>>> a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'f4')])
>>> a[['a', 'c']]
array([(0, 0.), (0, 0.), (0, 0.)],
     dtype={'names':['a','c'], 'formats':['<i4','<f4'], 'offsets':[0,8], 'itemsize':12})
>>> a[['a', 'c']] = (2, 3)
>>> a
array([(2, 0, 3.0), (2, 0, 3.0), (2, 0, 3.0)],
      dtype=[('a', '<i8'), ('b', '<i4'), ('c', '<f8')])

>>> x = np.array([(1, 2., 3.)], dtype='i,f,f')
>>> scalar = x[0]
>>> scalar
(1, 2., 3.)
>>> type(scalar)
numpy.void
#索引结构化数组的单个元素(带有整数索引)返回结构化标量

结构化标量还支持按字段名进行访问和赋值
>>> x = np.array([(1,2),(3,4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
>>> s = x[0]
>>> s['bar'] = 100
>>> x
array([(1, 100.), (3, 4.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])

结构化标量也可以用整数索引
>>> scalar = np.array([(1, 2., 3.)], dtype='i,f,f')[0]
>>> scalar[0]
1
>>> scalar[1] = 4

>>> scalar.item(), type(scalar.item())
((1, 2.0, 3.0), tuple)#结构化标量转换为元组

结构比较
>>> a = np.zeros(2, dtype=[('a', 'i4'), ('b', 'i4')])
>>> b = np.ones(2, dtype=[('a', 'i4'), ('b', 'i4')])
>>> a == b
array([False, False])#返回一个和原结构相同的bool数组

记录数组
numpy.rec.array它允许通过属性访问结构化数组的字段，而不仅仅是通过索引
>>> recordarr = np.rec.array([(1,2.,'Hello'),(2,3.,"World")],
...                    dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
>>> recordarr.bar
array([ 2.,  3.], dtype=float32)
>>> recordarr[1:2]
rec.array([(2, 3.0, 'World')],
      dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])
>>> recordarr[1:2].foo
array([2], dtype=int32)
>>> recordarr.foo[1:2]
array([2], dtype=int32)
>>> recordarr[1].baz
'World'

>>> arr = array([(1,2.,'Hello'),(2,3.,"World")],
...             dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
>>> recordarr = np.rec.array(arr)
#用np.rec.array转换结构化数据为记录数组


>>> arr = np.array([(1,2.,'Hello'),(2,3.,"World")],
... dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'a10')])
>>> arr
array([(1, 2.0, b'Hello'), (2, 3.0, b'World')],
      dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])
>>> arr.view()#查看
array([(1, 2.0, b'Hello'), (2, 3.0, b'World')],
      dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])
>>> rec_arr = arr.view(np.recarray)#查看类型的np.recarray的ndarray会转换为rec.record类型
>>> rec_arr
rec.array([(1, 2.0, b'Hello'), (2, 3.0, b'World')],
          dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])
>>> rec_arr.dtype
dtype((numpy.record, [('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')]))


























































