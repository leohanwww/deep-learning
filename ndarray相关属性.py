ndarray相关属性

ndarray.flags 有关数组内存分布的信息。
ndarray.shape 数组维度的元组。
ndarray.strides 遍历数组时要在每个维度中执行的字节元组。
ndarray.ndim 数组维数。
ndarray.data 指向数组数据开始的Python缓冲区对象。
ndarray.size 数组中的元素数。
ndarray.itemsize 一个数组元素的长度(以字节为单位)。
ndarray.nbytes 数组元素消耗的总字节。
ndarray.base 如果内存来自其他对象，则为基本对象。

数据类型
ndarray.T 与 self.transpose()相同，只是如果 self.ndim <2 则返回自己。
ndarray.real 数组的真实部分。
ndarray.imag 数组的虚部。
ndarray.flat 数组上的一维迭代器。
ndarray.ctypes 一个简化数组与ctypes模块交互的对象。


数组的方法
ndarray 对象有许多方法以某种方式对数组进行操作或与数组一起操作，通常返回数组结果。 下面简要说明这些方法。（每个方法的文档都有更完整的描述。）

对于以下方法，numpy中还有相应的函数：all，any，argmax，argmin，argpartition，argsort，choose，clip，compress，copy，cumprod，cumsum，diagonal，imag，max，mean，min，nonzero，partition， prod，ptp，put，ravel，real，repeat，reshape，round，searchsorted，sort，squeeze，std，sum，swapaxes，take，trace，transpose，var。


数组转换
ndarray.item(*args) 将数组元素复制到标准Python标量并返回它。
ndarray.tolist() 将数组作为（可能是嵌套的）列表返回。
ndarray.itemset(*args) 将标量插入数组（如果可能，将标量转换为数组的dtype）
ndarray.tostring([order]) 构造包含数组中原始数据字节的Python字节。
ndarray.tobytes([order]) 构造包含数组中原始数据字节的Python字节。
#    order : {'C', 'F', None}, optional
#        Order of the data for multidimensional arrays:
#        C, Fortran, or the same as for the original array.
ndarray.tofile(fid[, sep, format]) 将数组作为文本或二进制写入文件（默认）。
ndarray.dump(file) 将数组的pickle转储到指定的文件。
ndarray.dumps() 以字符串形式返回数组的pickle。
ndarray.astype(dtype[, order, casting, …]) 数组的副本，强制转换为指定的类型。
ndarray.byteswap([inplace]) 交换数组元素的字节
ndarray.copy([order]) 返回数组的副本。
ndarray.view([dtype, type]) 具有相同数据的数组的新视图。
ndarray.getfield(dtype[, offset]) 返回给定数组的字段作为特定类型。
ndarray.setflags([write, align, uic]) 分别设置数组标志WRITEABLE，ALIGNED，（WRITEBACKIFCOPY和UPDATEIFCOPY）。
ndarray.fill(value) 使用标量值填充数组。

项目选择和操作
对于采用axis关键字的数组方法，默认为None。 如果axis为None，则将数组视为1维数组。轴的任何其他值表示操作应该沿着的维度。
ndarray.take(indices[, axis, out, mode]) 返回由给定索引处的a元素组成的数组。
ndarray.put(indices, values[, mode]) 为索引中的所有n设置 a.flat[n] = values[n]。
ndarray.repeat(repeats[, axis]) 重复数组的元素。axis=0是行，axis=1是列
ndarray.choose(choices[, out, mode]) 使用索引数组从一组选项中构造新数组。
ndarray.sort([axis, kind, order]) 就地对数组进行排序。
ndarray.argsort([axis, kind, order]) 返回将对此数组进行排序的索引。
ndarray.partition(kth[, axis, kind, order]) 重新排列数组中的元素，使得第k个位置的元素值处于排序数组中的位置。
ndarray.argpartition(kth[, axis, kind, order]) 重新排列数组中的元素，使得第k个位置的元素值处于排序数组中的位置。
ndarray.searchsorted(v[, side, sorter]) 查找应在其中插入v的元素以维护顺序的索引。
ndarray.nonzero() 返回非零元素的索引。
ndarray.compress(condition[, axis, out]) 沿给定轴返回此数组的选定切片。
ndarray.diagonal([offset, axis1, axis2]) 返回指定的对角线。

计算

>>> x #在3维数组的其中一轴上求和，将返回两维的数组
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],
       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],
       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
>>> x.sum(axis=0)
array([[27, 30, 33],
       [36, 39, 42],
       [45, 48, 51]])
>>> x.sum(0), x.sum(1), x.sum(2)
(array([[27, 30, 33],
        [36, 39, 42],
        [45, 48, 51]]),
 array([[ 9, 12, 15],
        [36, 39, 42],
        [63, 66, 69]]),
 array([[ 3, 12, 21],
        [30, 39, 48],
        [57, 66, 75]]))

ndarray.argmax([axis, out]) 返回给定轴的最大值索引。
ndarray.min([axis, out, keepdims]) 沿给定轴返回最小值。
ndarray.argmin([axis, out]) 沿a的给定轴返回最小值的索引。
ndarray.ptp([axis, out]) 沿给定轴的峰峰值（最大值 - 最小值）。
ndarray.clip([min, max, out]) 返回其值限制为 [min, max] 的数组。
ndarray.conj() 复合共轭所有元素。
ndarray.round([decimals, out]) 返回a，每个元素四舍五入到给定的小数位数。
ndarray.trace([offset, axis1, axis2, dtype, out]) 返回数组对角线的总和。
ndarray.sum([axis, dtype, out, keepdims]) 返回给定轴上的数组元素的总和。
ndarray.cumsum([axis, dtype, out]) 返回给定轴上元素的累积和。
ndarray.mean([axis, dtype, out, keepdims]) 返回给定轴上数组元素的平均值。
ndarray.var([axis, dtype, out, ddof, keepdims]) 返回给定轴的数组元素的方差。
ndarray.std([axis, dtype, out, ddof, keepdims]) 返回给定轴的数组元素的标准偏差。
ndarray.prod([axis, dtype, out, keepdims]) 返回给定轴上的数组元素的乘积
ndarray.cumprod([axis, dtype, out]) 返回沿给定轴的元素的累积乘积。
ndarray.all([axis, out, keepdims]) 如果所有元素都计算为True，则返回True。
ndarray.any([axis, out, keepdims]) 如果求值的任何元素为True，则返回True。


比较运算符:
ndarray.lt($self, value, /) 返回 self
ndarray.le($self, value, /) 返回 self<=value.
ndarray.gt($self, value, /) 返回 self>value.
ndarray.ge($self, value, /) 返回 self>=value.
ndarray.eq($self, value, /) 返回 self==value.
ndarray.ne($self, value, /) 返回 self!=value.

一元操作：
ndarray.neg($self, /) -self
ndarray.pos($self, /) +self
ndarray.abs(self)
ndarray.invert($self, /) ~self

算术运算：
ndarray.add($self, value, /) 返回 self+value. #只能用np.add(a,b)
ndarray.sub($self, value, /) 返回 self-value.
ndarray.mul($self, value, /) 返回 self*value.
ndarray.div
ndarray.truediv($self, value, /) 返回 self/value.
ndarray.floordiv($self, value, /) 返回 self//value.
ndarray.mod($self, value, /) 返回 self%value.
ndarray.divmod($self, value, /) 返回 divmod(self, value).
ndarray.pow($self, value) 返回 pow(self, value).
ndarray.lshift($self, value, /) 返回 self<
ndarray.rshift($self, value, /) 返回 self>>value.
ndarray.and($self, value, /) 返回 self&value.
ndarray.or($self, value, /) 返回 self|value.
ndarray.xor($self, value, /) 返回 self^value.

就地算数运算：
ndarray.iadd($self, value, /) 返回 self+=value.
ndarray.isub($self, value, /) 返回 self-=value.
ndarray.imul($self, value, /) 返回 self*=value.
ndarray.idiv
ndarray.itruediv($self, value, /) 返回 self/=value.
ndarray.ifloordiv($self, value, /) Return self//=value.
ndarray.imod($self, value, /) 返回 self%=value.
ndarray.ipow($self, value, /) 返回 self**=value.
ndarray.ilshift($self, value, /) 返回 self<<=value.
ndarray.irshift($self, value, /) 返回 self>>=value.
ndarray.iand($self, value, /) 返回 self&=value.
ndarray.ior($self, value, /) 返回 self|=value.
ndarray.ixor($self, value, /) 返回 self^=value.

标准库函数：
ndarray.copy() 如果在数组上调用copy.copy，则使用此方法。
ndarray.deepcopy(memo, /) 如果在数组上调用copy.deepcopy，则使用此方法。
ndarray.reduce() 用于腌制（译者注：很形象）。
ndarray.setstate(state, /) 用于反腌制。
ndarray.new($type, args, *kwargs) 创建并返回一个新对象。
ndarray.array(|dtype) 如果没有给出dtype，则返回对self的新引用;如果dtype与数组的当前dtype不同，则返回提供的数据类型的新数组。
ndarray.array_wrap(obj)

ndarray.len($self, /) 返回 len(self).
ndarray.getitem($self, key, /) 返回 self[key].
ndarray.setitem($self, key, value, /) 给 self[key] 设置一个值。
ndarray.contains($self, key, /) 返回 自身的关键索引。

转换;操作complex，int，long，float，oct和hex。它们位于数组中，其中包含一个元素并返回相应的标量。
ndarray.int(self)#
ndarray.long
ndarray.float(self)
ndarray.oct
ndarray.hex

字符串表示：
ndarray.str($self, /) 返回 str(self).#np.str(a)
ndarray.repr($self, /) 返回 repr(self).

浮点类型
half	| 'e'	
single	兼容: C float	'f'
double	兼容: C double	
float_	兼容: Python float	'd'
longfloat	兼容: C long float	'g'
float16	16 bits	
float32	32 bits	
float64	64 bits	
float96	96 bits, platform?	
float128	128 bits, platform?	

复杂的浮点数：

类型	备注	字符代码
csingle	| 'F'	
complex_	兼容: Python complex 类型	'D'
clongfloat	| 'G'	
complex64	两个 32-bit 浮点数	
complex128	两个 64-bit 浮点数	
complex192	两个 96-bit 浮点数, platform?	
complex256	两个 128-bit 浮点数, platform?	

任意Python对象：

类型	备注	字符代码
object_	一个 Python 对象	'O'













































































































