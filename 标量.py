标量

NumPy中的默认数据类型是float_

布尔值：

类型		备注						字符代码
bool_	compatible: Python bool	'?'
bool8	8 bits	

整型
类型							字符代码
byte	compatible: C char	'b'
short	compatible: C short	'h'
intc	compatible: C int	'i'
int_	compatible: Python int	'l'
longlong	compatible: C long long	'q'
intp	large enough to fit a pointer	'p'
int8	8 bits	
int16	16 bits	
int32	32 bits	
int64	64 bits
无符号整形
ubyte	兼容: C unsigned char	'B'
ushort	兼容: C unsigned short	'H'
uintc	兼容: C unsigned int	'I'
uint	兼容: Python int	'L'
ulonglong	兼容: C long long	'Q'
uintp	大到足以适合指针	'P'
uint8	8 bits	-
uint16	16 bits	-
uint32	32 bits	-
uint64	64 bits	-


数据类型对象

>>> dt = np.dtype('>i4')
>>> dt.byteorder
'>'
>>> dt.itemsize
4
>>> dt.name
'int32'
>>> dt.type is np.int32
True

>>> dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
>>> dt['name']
dtype('|U16')
>>> dt['grades']
dtype(('float64',(2,)))
>>> x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
>>> x[1]
('John', [6.0, 7.0])
>>> x[1]['grades']
array([ 6.,  7.])
>>> type(x[1])
<type 'numpy.void'>
>>> type(x[1]['grades'])
<type 'numpy.ndarray'>

具有dtype属性的任何类型对象：将直接访问和使用该属性。 该属性必须返回可转换为dtype对象的内容。
可以转换几种字符串。 可以使用'>'（big-endian），'<'（little-endian）或'='（hardware-native, the default）来预先识别字符串，以指定字节顺序。
Little-Endian就是低位字节排放在内存的低地址端，高位字节排放在内存的高地址端。
Big-Endian就是高位字节排放在内存的低地址端，低位字节排放在内存的高地址端。
>>> dt = np.dtype('b')  # byte, native byte order
>>> dt = np.dtype('>H') # big-endian unsigned short
>>> dt = np.dtype('<f') # little-endian single-precision float
>>> dt = np.dtype('d')  # double-precision floating-point number

第一个字符指定数据类型，其余字符指定每个项目的字节数，Unicode除外，其中它被解释为字符数。项目大小必须与现有类型相对应，否则将引发错误。支持的种类是：
'?'	boolean
'b'	(signed) byte
'B'	unsigned byte
'i'	(signed) integer
'u'	unsigned integer
'f'	floating-point
'c'	complex-floating point
'm'	timedelta
'M'	datetime
'O'	(Python) objects
'S', 'a'	zero-terminated bytes (not recommended)
'U'	Unicode string
'V'	raw data (void)

>>> dt = np.dtype('i4')   # 32-bit signed integer
>>> dt = np.dtype('f8')   # 64-bit floating-point number
>>> dt = np.dtype('c16')  # 128-bit complex floating-point number
>>> dt = np.dtype('a25')  # 25-length zero-terminated bytes
>>> dt = np.dtype('U25')  # 25-character string

>>> dt = np.dtype('uint32')   # 32-bit unsigned integer
>>> dt = np.dtype('Float64')  # 64-bit floating-point number

>>> dt = np.dtype((np.void, 10))  # 10-byte wide data block 第二个参数是itemsize
>>> dt = np.dtype(('U', 10))   # 10-character unicode string

>>> dt = np.dtype((np.int32, (2,2)))          # 2 x 2 integer sub-array
>>> dt = np.dtype(('U10', 1))                 # 10-character string
>>> dt = np.dtype(('i4, (2,3)f8, f4', (2,3))) # 2 x 3 structured sub-array

>>> dt = np.dtype({'names': ['r','g','b','a'],
...                'formats': [uint8, uint8, uint8, uint8]})

>>> dt = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
...                'offsets': [0, 2],
...                'titles': ['Red pixel', 'Blue pixel']})

>>> dt = np.dtype({'col1': ('U10', 0), 'col2': (float32, 10),
    'col3': (int, 14)})
#位于0字节处的10个字符的字符串


32位整数，被解释为由包含8位整数的形状 (4, ) 的子数组组成：
>>> dt = np.dtype((np.int32, (np.int8, 4)))

32位整数，包含字段r，g，b，a，将整数中的4个字节解释为四个无符号整数：
>>> dt = np.dtype(('i4', [('r','u1'),('g','u1'),('b','u1'),('a','u1')]))


















































































