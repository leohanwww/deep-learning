子类化ndarray

视图投影
视图投影是标准的ndarray机制，通过它你可以获取任何子类的ndarray，并将该数组的视图作为另一个（指定的）子类返回
>>> import numpy as np
>>> class C(np.ndarray): pass
>>> arr = np.zeros((3,))
>>> c_arr = arr.view(C)
>>> type(c_arr)
<class 'C'>

从模版创建
当numpy发现它需要从模板实例创建新实例时，ndarray子类的新实例也可以通过与视图转换非常相似的机制来实现。 这个情况的最明显的时候是你正为子类阵列切片的时候。
>>> v = c_arr[1:]
>>> type(v) # the view is of type 'C'
<class 'C'>
>>> v is c_arr # but it's a new instance
False
#切片是原始C_ARR数据的视图。因此，当我们从ndarray获取视图时，我们返回一个新的ndarray，它属于同一个类，指向原始的数据。

ndarray使用__new__方法来创建新的实例，但是ndarray__new__方法不知道我们在自己的__new__方法中做了什么来设置属性
__array_finalize__是numpy提供的机制，允许子类处理创建新实例的各种方法

import numpy as np

class C(np.ndarray):
    def __new__(cls, *args, **kwargs):
        print('In __new__ with class %s' % cls)
        return super(C, cls).__new__(cls, *args, **kwargs)
#可以看到super调用，它转到ndarray .__new__，将__array_finalize__传递给我们自己的类（self）
    def __init__(self, *args, **kwargs):
        # in practice you probably will not need or want an __init__
        # method for your subclass
        print('In __init__ with class %s' % self.__class__)

    def __array_finalize__(self, obj):
        print('In array_finalize:')
        print('   self type is %s' % type(self))
        print('   obj type is %s' % type(obj))

>>> # Explicit constructor
>>> c = C((10,))
In __new__ with class <class 'C'>
In array_finalize:
   self type is <class 'C'>
   obj type is <type 'NoneType'>#当从显式构造函数调用时，“obj”是“None”
In __init__ with class <class 'C'>
>>> # View casting
>>> a = np.arange(10)
>>> cast_a = a.view(C)
In array_finalize:
   self type is <class 'C'>
   obj type is <type 'numpy.ndarray'>#当从视图转换调用时，obj可以是ndarray的任何子类的实例，包括我们自己的子类。
>>> # Slicing (example of new-from-template)
>>> cv = c[:1]
In array_finalize:
   self type is <class 'C'>
   obj type is <class 'C'>#当在新模板中调用时，obj是我们自己子类的另一个实例，我们可以用它来更新的self实例。

使用_finalize_是始终能看到创建新实例的方法，其余方法不能显式调用

向ndarray添加额外属性

import numpy as np

class RealisticInfoArray(np.ndarray):
#继承标准ndarray，并转换为自定义的类型
    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)

>>> arr = np.arange(5)
>>> obj = RealisticInfoArray(arr, info='information')
>>> type(obj)
<class 'RealisticInfoArray'>
>>> obj.info
'information'
>>> v = obj[1:]
>>> type(v)
<class 'RealisticInfoArray'>
>>> v.info
'information'



















































































































































































































