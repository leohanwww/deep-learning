字节排序和ndarrays
ndarray 是为内存中的数据提供python数组接口的对象。

>>> big_end_str = chr(0) + chr(1) + chr(3) + chr(2)
>>> big_end_str
'\x00\x01\x03\x02'
>>> big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=bytes(big_end_str, encoding='utf8'))
>>> big_end_arr
array([  1, 770], dtype=int16)
#>表示'big-endian'（<是小端），i2表示'带符号的2字节整数'
>>> little_end_u4 = np.ndarray(shape=(1,),dtype='<u4', buffer=big_end_str)
>>> little_end_u4[0] == 1 * 256**1 + 3 * 256**2 + 2 * 256**3
True
>>> little_end_u4
array([33751296], dtype=uint32)

#更改数组dtype中的字节顺序信息，以便它将未确定的数据解释为处于不同的字节顺序。这是arr.newbyteorder()的作用
#改变底层数据的字节顺序，保持原来的dtype解释。这是arr.byteswap()所做的。

数据和dtype字节顺序不匹配，将dtype更改为匹配数据
>>> wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_str)
>>> wrong_end_dtype_arr[0]
256
>>> fixed_end_dtype_arr = wrong_end_dtype_arr.newbyteorder()
#使用newbyteorder重新排序
>>> fixed_end_dtype_arr[0]
1
>>> fixed_end_dtype_arr.tobytes() == big_end_str #数组在内存中没有改变
True

数据和类型字节顺序不匹配，更改数据以匹配dtype
>>> fixed_end_mem_arr = wrong_end_dtype_arr.byteswap()
>>> fixed_end_mem_arr[0]
1
>>> fixed_end_mem_arr.tobytes() == big_end_str #数组在内存里改变了
False

数据和dtype字节顺序匹配，交换数据和dtype
>>> swapped_end_arr = big_end_arr.byteswap().newbyteorder()
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_str
False
>>> swapped_end_arr = big_end_arr.astype('<i2')
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_str
False


使用ndarray astype方法可以实现将数据转换为特定dtype和字节顺序的更简单的方法：
>>> swapped_end_arr = big_end_arr.astype('<i2')
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_str
False


































































































