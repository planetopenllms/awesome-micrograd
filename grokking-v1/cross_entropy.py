from mintorch import Tensor, nn, _np_softmax, _np_onehot, _np_softmax_cross_entropy

import numpy as np


#####
## use low-level methods

#  (batch x classes)  logits
y_hat = np.array( [[4,3,2,1],
                   [-1,2,5,1],
                   [0,1,2,3]]
                  )

print( "y_hat:", y_hat )
print( "softmax:", _np_softmax( y_hat ))


# target class indices (no need to hot-encode)
#   (classes)  indices
y = np.array([0,1,3])
print( "y:", y )

classes_size = y_hat.shape[1]
print( "onehot:", _np_onehot( y, classes=classes_size))

print( "softmax_cross_entropy:", _np_softmax_cross_entropy( y_hat, y ))

print( "---")
###
##   numerical stability  (e.g. try e**x with x=1000)
y_hat = np.array( [[400,300,200,100],
                   [-1000,2000,5000,1000],
                   [0,100,200,3]]
                  )

print( "y_hat:", y_hat )
print( "softmax:", _np_softmax( y_hat ))
print( "softmax_cross_entropy:", _np_softmax_cross_entropy( y_hat, y ))



#####
## use high-level methods
print( "--- high-level:" )

criterion = nn.CrossEntropyLoss()

y_hat = Tensor(y_hat, requires_grad=True)
y     = Tensor(y)
loss = criterion( y_hat, y )
print( "loss:", loss )

loss.backward()
print( "y_hat", y_hat )
print( "y_hat.grad", y_hat.grad )



print( "bye" )
