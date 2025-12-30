from mintorch import Tensor, nn  

##
#   (batch x classes)
#   --
#   (target_indices) - same as (batch x classes)

# data indices
#  (xclasses)
y_hat = Tensor( [[4,3,2,1],
                  [-1,2,5,1],
                  [0,1,2,3]
                  ], requires_grad=True)

# target class indices (no need to hot-encode)
y = Tensor([0,1,3])

print( y_hat )
print( y )


criterion = nn.CrossEntropyLoss()

loss = criterion( y_hat, y )

# loss = y_hat.cross_entropy( y ) 
print( loss )

loss.backward()
print( "y_hat.grad", y_hat.grad )


#############
## check with batch dim
print( "\n=== with batch dim" )

##
#   (batch x n_tokens x classes)
#   --
#   (n_tokens x target_indices) - same as (batch x n_tokens x classes)

y_hat = Tensor( [[[4,3,2,1],
                  [-1,2,5,1],
                  [0,1,2,3]],
                  [[1,2,3,4],
                   [-1,2,8,2],
                   [3,2,1,0]]
                  ], requires_grad=True)

# target indices
y = Tensor([[0,1,3],
            [3,1,0]])

print( y_hat )
print( y )

loss = criterion( y_hat, y )

# loss = y_hat.cross_entropy( y ) 
print( loss )

loss.backward()
print( "y_hat.grad", y_hat.grad )



print("bye")


"""
y_hat = Tensor( [[0.3,0.7],
                 [0.1,0.9]
                 ], requires_grad=True)
y = Tensor([[0,1],
            [1,0]])
"""
