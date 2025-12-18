import numpy as np


def p(obj):
   print( f"{obj}, shape={obj.shape}, ndim={obj.ndim}, size={obj.size}, dtype={obj.dtype}" )

## check convert to ndarray

## pass-in python list
a = np.array( [1,2,3,4], dtype=np.float32 )
p( a )

aa = np.array( [[1,2,3,4],
                [5,6,7,8]], dtype=np.float32 )
p( aa )
 
## pass-in ndarray
b = np.array( a )
p( b )

b = np.array( np.array( [1,2,3,4]), dtype=np.float32 )

## check scalars
c = np.array( 1 )
p( c )

d = np.array( [1] )
p( d )


print( "bye" )