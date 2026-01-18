## plot some math functions
##   e.g.   e^x, e^-x,  sigmoid etc.


import matplotlib.pyplot as plt
import numpy as np

x = np.array( [-5,-3,-2,-1,-0.99,-0.5,-0.1,-0.01,0,
               0.01,0.1,0.5,0.99,1,2,3,5] )

y = np.exp( x )

print("x=",x)
print("y=",y)

y2 = np.exp( -x )
print("y2=",y2)

## sigmoid
y3 = 1 / (1 + np.exp( -x ))
print( "y3=", y3)
### sigmoid deriv    - sigmoid * (1-sigmoid)
y4 = y3 * (1-y3)
print( "y4=", y4 )

plt.step( x, y )
plt.plot( x, y )
plt.title( "e^x" )
plt.show()


plt.step( x, y2 )
plt.plot( x, y2 )
plt.title( "e^-x" )
plt.show()


plt.plot( x, y )
plt.plot( x, y2 )
plt.title( "e^x, e^-x" )
plt.show()



plt.step( x, y3, color="blue" )
plt.plot( x, y3, color="red" )
plt.plot( x, y4, color="green" )   ## add sigmoid_deriv
plt.title( "1/(1+e^-x)" )
plt.show()


print("bye")