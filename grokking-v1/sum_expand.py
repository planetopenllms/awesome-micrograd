from mintorch import Tensor, _np_expand


x = Tensor([[1,2,3],
           [4,5,6]])
print(x.sum(dim=0))
# array([5, 7, 9])
print(x.sum(dim=1))
# array([ 6, 15])


print(x.expand(dim=2, copies=4))
# array([[[1, 1, 1, 1],
#        [2, 2, 2, 2],
#        [3, 3, 3, 3]],
#        [[4, 4, 4, 4],
#        [5, 5, 5, 5],
#        [6, 6, 6, 6]]])


print( "--- dim=2" )
xx = _np_expand(x.data, dim=2, copies=4)
print(x.data.shape, x.data)
print(xx.shape, xx)

print( "--- dim=0" )
xx = _np_expand(x.data, dim=0, copies=4)
print(x.data.shape, x.data)
print(xx.shape, xx)


print("bye")
