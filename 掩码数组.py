掩码数组

>>> import numpy as np
>>> import numpy.ma as ma
>>> x = np.array([1, 2, 3, -1, 5])
>>> mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0])
>>> mx.mean()
2.75

使用现有数组构造
>>> x = np.array([1, 2, 3])
>>> x.view(ma.MaskedArray)
masked_array(data = [1 2 3],
              mask = False,
      fill_value = 999999)
>>> x = np.array([(1, 1.), (2, 2.)], dtype=[('a',int), ('b', float)])
>>> x.view(ma.MaskedArray)
masked_array(data = [(1, 1.0) (2, 2.0)],
              mask = [(False, False) (False, False)],
      fill_value = (999999, 1e+20),
              dtype = [('a', '<i4'), ('b', '<f8')])


>>> x = ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
>>> x[~x.mask]或x[x.mask]
masked_array(data = [1 4],
             mask = [False False],
       fill_value = 999999)
>>> x.compressed()#只查看有效数据
array([1, 4])


>>> x = ma.array([1, 2, 3])
>>> x[0] = ma.masked#手动掩盖元素
>>> x
masked_array(data = [-- 2 3],
             mask = [ True False False],
       fill_value = 999999)
>>> y = ma.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> y[(0, 1, 2), (1, 2, 0)] = ma.masked
>>> y
masked_array(data =
 [[1 -- 3]
  [4 5 --]
  [-- 8 9]],
             mask =
 [[False  True False]
  [False False  True]
  [ True False False]],
       fill_value = 999999)
>>> z = ma.array([1, 2, 3, 4])
>>> z[:-2] = ma.masked
>>> z
masked_array(data = [-- -- 3 4],
             mask = [ True  True False False],
       fill_value = 999999)

>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x.mask = True#全部掩盖
>>> x
masked_array(data = [-- -- --],
             mask = [ True  True  True],
       fill_value = 999999)
>>> x = ma.array([1, 2, 3])
>>> x.mask = [0, 1, 0]
>>> x
masked_array(data = [1 -- 3],
             mask = [False  True False],
       fill_value = 999999)

取消屏蔽条目
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x
masked_array(data = [1 2 --],
             mask = [False False  True],
       fill_value = 999999)
>>> x[-1] = 5#直接给屏蔽条目分配一个有效值就可以
>>> x
masked_array(data = [1 2 5],
             mask = [False False False],
       fill_value = 999999)

>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x
masked_array(data = [1 2 --],
             mask = [False False  True],
       fill_value = 999999)
>>> x.mask = ma.nomask#取消整个掩码
>>> x
masked_array(data = [1 2 3],
             mask = [False False False],
       fill_value = 999999)

>>> y
masked_array(data = [(1, 2) (3, --)],
             mask = [(False, False) (False, True)],
       fill_value = (999999, 999999),
            dtype = [('a', '<i4'), ('b', '<i4')])
>>> y['a']
masked_array(data = [1 3],
             mask = [False False],
       fill_value = 999999)
>>> y['b']
masked_array(data = [2 --],
             mask = [False  True],
       fill_value = 999999)
>>> y['b'][1]
masked

掩码的切片
>>> x = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 1])
>>> mx = x[:3]
>>> mx
masked_array(data = [1 -- 3],
             mask = [False  True False],
       fill_value = 999999)
>>> mx[1] = -1
>>> mx
masked_array(data = [1 -1 3],
             mask = [False False False],
       fill_value = 999999)
>>> x.mask
array([False,  True, False, False,  True])
>>> x.data
array([ 1, -1,  3,  4,  5])

>>> import numpy.ma as ma
>>> x = [0.,1.,-9999.,3.,4.]
>>> mx = ma.masked_values (x, -9999.)
>>> print mx.mean()
2.0
>>> print mx - mx.mean()
[-2.0 -1.0 -- 1.0 2.0]
>>> print mx.anom()
[-2.0 -1.0 -- 1.0 2.0]

>>> print mx.filled(mx.mean())#用平均值填充缺失数据
[ 0.  1.  2.  3.  4.]


>>> x = ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
...        dtype=[('a', int), ('b', int)])
>>> x.recordmask
array([False, False,  True, False, False])

MaskedArray.fill_value
数据类型	默认值
bool	True
int	999999
float	1.e20
complex	1.e20+0j
object	‘?’
string	‘N/A’









































































































































































