###
##
#   note:
#     maybe dropout backprob is not 100% correct
#       2 = 1/(1-0.5)
#       dropout_mask * 2
#    why not dropout_mask inline e.g.
#       dropout_mask = dropout_mask*2
#    and than in backward
#       dropout_mask "automagically" includes scaling factor (1/1-p)


import sys, numpy as np
import mnist

(images, labels), (test_images, test_labels) = mnist.load_data()


np.random.seed(1)
def relu(x):
    return (x >= 0) * x # returns x if x > 0
                        # returns 0 otherwise

def relu2deriv(output):
    return output >= 0 #returns 1 for input > 0

alpha, iterations, hidden_size = (0.005, 300, 100)
pixels_per_image, num_labels = (784, 10)

weights_0_1 = 0.2*np.random.random((pixels_per_image,hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size,num_labels)) - 0.1

for j in range(iterations):
    error, correct_cnt = (0.0,0)
    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1,weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))
        layer_2_delta = (labels[i:i+1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if(j%10 == 0):
        test_error, test_correct_cnt = (0.0,0)
         
        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = relu(np.dot(layer_0,weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            test_error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
            test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

        sys.stdout.write("\n" + \
                         "I:" + str(j) + \
                         " Test-Err:" + str(test_error/ float(len(test_images)))[0:5] +\
                         " Test-Acc:" + str(test_correct_cnt/ float(len(test_images)))+\
                         " Train-Err:" + str(error/ float(len(images)))[0:5] +\
                         " Train-Acc:" + str(correct_cnt/ float(len(images))))
        

print("\nbye")


"""
I:0 Test-Err:0.641 Test-Acc:0.6333 Train-Err:0.891 Train-Acc:0.413
I:10 Test-Err:0.458 Test-Acc:0.787 Train-Err:0.472 Train-Acc:0.764
I:20 Test-Err:0.415 Test-Acc:0.8133 Train-Err:0.430 Train-Acc:0.809
I:30 Test-Err:0.421 Test-Acc:0.8114 Train-Err:0.415 Train-Acc:0.811
I:40 Test-Err:0.419 Test-Acc:0.8112 Train-Err:0.413 Train-Acc:0.827
I:50 Test-Err:0.409 Test-Acc:0.8133 Train-Err:0.392 Train-Acc:0.836
I:60 Test-Err:0.412 Test-Acc:0.8236 Train-Err:0.402 Train-Acc:0.836
I:70 Test-Err:0.412 Test-Acc:0.8033 Train-Err:0.383 Train-Acc:0.857
I:80 Test-Err:0.410 Test-Acc:0.8054 Train-Err:0.386 Train-Acc:0.854
I:90 Test-Err:0.411 Test-Acc:0.8144 Train-Err:0.376 Train-Acc:0.868
I:100 Test-Err:0.411 Test-Acc:0.7903 Train-Err:0.369 Train-Acc:0.864
I:110 Test-Err:0.411 Test-Acc:0.8003 Train-Err:0.371 Train-Acc:0.868
I:120 Test-Err:0.402 Test-Acc:0.8046 Train-Err:0.353 Train-Acc:0.857
I:130 Test-Err:0.408 Test-Acc:0.8091 Train-Err:0.352 Train-Acc:0.867
I:140 Test-Err:0.405 Test-Acc:0.8083 Train-Err:0.355 Train-Acc:0.885
I:150 Test-Err:0.404 Test-Acc:0.8107 Train-Err:0.342 Train-Acc:0.883
I:160 Test-Err:0.399 Test-Acc:0.8146 Train-Err:0.361 Train-Acc:0.876
I:170 Test-Err:0.404 Test-Acc:0.8074 Train-Err:0.344 Train-Acc:0.889
I:180 Test-Err:0.399 Test-Acc:0.807 Train-Err:0.333 Train-Acc:0.892
I:190 Test-Err:0.407 Test-Acc:0.8066 Train-Err:0.335 Train-Acc:0.898
I:200 Test-Err:0.405 Test-Acc:0.8036 Train-Err:0.347 Train-Acc:0.893
I:210 Test-Err:0.405 Test-Acc:0.8034 Train-Err:0.336 Train-Acc:0.894
I:220 Test-Err:0.402 Test-Acc:0.8067 Train-Err:0.325 Train-Acc:0.896
I:230 Test-Err:0.404 Test-Acc:0.8091 Train-Err:0.321 Train-Acc:0.894
I:240 Test-Err:0.415 Test-Acc:0.8091 Train-Err:0.332 Train-Acc:0.898
I:250 Test-Err:0.395 Test-Acc:0.8182 Train-Err:0.320 Train-Acc:0.899
I:260 Test-Err:0.390 Test-Acc:0.8204 Train-Err:0.321 Train-Acc:0.899
I:270 Test-Err:0.382 Test-Acc:0.8194 Train-Err:0.312 Train-Acc:0.906
I:280 Test-Err:0.396 Test-Acc:0.8208 Train-Err:0.317 Train-Acc:0.9
I:290 Test-Err:0.399 Test-Acc:0.8181 Train-Err:0.301 Train-Acc:0.908

vs
using 
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        dropout_mask *= 2
        layer_1 *= dropout_mask 
          instead of
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        note - difference is in backprop!

I:0 Test-Err:0.625 Test-Acc:0.6522 Train-Err:0.842 Train-Acc:0.456
I:10 Test-Err:0.445 Test-Acc:0.7872 Train-Err:0.442 Train-Acc:0.797
I:20 Test-Err:0.413 Test-Acc:0.8129 Train-Err:0.402 Train-Acc:0.838
I:30 Test-Err:0.407 Test-Acc:0.8211 Train-Err:0.393 Train-Acc:0.843
I:40 Test-Err:0.411 Test-Acc:0.8126 Train-Err:0.386 Train-Acc:0.846
I:50 Test-Err:0.413 Test-Acc:0.804 Train-Err:0.384 Train-Acc:0.842
I:60 Test-Err:0.415 Test-Acc:0.8057 Train-Err:0.368 Train-Acc:0.849
I:70 Test-Err:0.411 Test-Acc:0.7993 Train-Err:0.360 Train-Acc:0.877
I:80 Test-Err:0.403 Test-Acc:0.8092 Train-Err:0.352 Train-Acc:0.884
I:90 Test-Err:0.413 Test-Acc:0.8045 Train-Err:0.360 Train-Acc:0.869
I:100 Test-Err:0.410 Test-Acc:0.7918 Train-Err:0.345 Train-Acc:0.888
I:110 Test-Err:0.409 Test-Acc:0.7951 Train-Err:0.340 Train-Acc:0.88
I:120 Test-Err:0.409 Test-Acc:0.7896 Train-Err:0.340 Train-Acc:0.891
I:130 Test-Err:0.412 Test-Acc:0.782 Train-Err:0.330 Train-Acc:0.886
I:140 Test-Err:0.406 Test-Acc:0.8009 Train-Err:0.342 Train-Acc:0.889
I:150 Test-Err:0.423 Test-Acc:0.7859 Train-Err:0.322 Train-Acc:0.89
I:160 Test-Err:0.411 Test-Acc:0.7906 Train-Err:0.333 Train-Acc:0.884
I:170 Test-Err:0.414 Test-Acc:0.7867 Train-Err:0.324 Train-Acc:0.897
I:180 Test-Err:0.412 Test-Acc:0.7893 Train-Err:0.317 Train-Acc:0.902
I:190 Test-Err:0.411 Test-Acc:0.7966 Train-Err:0.316 Train-Acc:0.906
I:200 Test-Err:0.418 Test-Acc:0.7939 Train-Err:0.334 Train-Acc:0.898
I:210 Test-Err:0.431 Test-Acc:0.7795 Train-Err:0.313 Train-Acc:0.899
I:220 Test-Err:0.432 Test-Acc:0.7877 Train-Err:0.311 Train-Acc:0.913
I:230 Test-Err:0.431 Test-Acc:0.7915 Train-Err:0.305 Train-Acc:0.914
I:240 Test-Err:0.434 Test-Acc:0.7924 Train-Err:0.298 Train-Acc:0.908
I:250 Test-Err:0.421 Test-Acc:0.793 Train-Err:0.298 Train-Acc:0.915
I:260 Test-Err:0.422 Test-Acc:0.793 Train-Err:0.298 Train-Acc:0.919
I:270 Test-Err:0.412 Test-Acc:0.7936 Train-Err:0.300 Train-Acc:0.907
I:280 Test-Err:0.415 Test-Acc:0.7864 Train-Err:0.307 Train-Acc:0.903
I:290 Test-Err:0.426 Test-Acc:0.7859 Train-Err:0.290 Train-Acc:0.917
"""