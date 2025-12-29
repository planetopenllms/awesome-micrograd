

import sys, numpy as np
import mnist

(images, labels), (test_images, test_labels) = mnist.load_data()


np.random.seed(1)
def relu(x):
    return (x >= 0) * x # returns x if x > 0

def relu2deriv(output):
    return output >= 0 # returns 1 for input > 0

batch_size = 100
alpha, iterations = (0.1, 300)
pixels_per_image, num_labels, hidden_size = (784, 10, 100)

weights_0_1 = 0.2*np.random.random((pixels_per_image,hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size,num_labels)) - 0.1

for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size),((i+1)*batch_size))

        layer_0 = images[batch_start:batch_end]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        dropout_mask = np.random.randint(2,size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1,weights_1_2)

        error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k+1]) == np.argmax(labels[batch_start+k:batch_start+k+1]))

        layer_2_delta = (labels[batch_start:batch_end]-layer_2) / batch_size
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)* relu2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
            
    if(j%10 == 0):
        test_error, test_correct_cnt = (0.0, 0)

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


print( "\nbye")

"""
I:0 Test-Err:0.815 Test-Acc:0.3832 Train-Err:1.272 Train-Acc:0.161
I:10 Test-Err:0.569 Test-Acc:0.7183 Train-Err:0.591 Train-Acc:0.672
I:20 Test-Err:0.508 Test-Acc:0.7577 Train-Err:0.530 Train-Acc:0.727
I:30 Test-Err:0.483 Test-Acc:0.7815 Train-Err:0.497 Train-Acc:0.758
I:40 Test-Err:0.464 Test-Acc:0.7915 Train-Err:0.486 Train-Acc:0.75
I:50 Test-Err:0.453 Test-Acc:0.7978 Train-Err:0.462 Train-Acc:0.784
I:60 Test-Err:0.446 Test-Acc:0.8015 Train-Err:0.445 Train-Acc:0.801
I:70 Test-Err:0.437 Test-Acc:0.8054 Train-Err:0.444 Train-Acc:0.807
I:80 Test-Err:0.440 Test-Acc:0.807 Train-Err:0.450 Train-Acc:0.803
I:90 Test-Err:0.437 Test-Acc:0.8059 Train-Err:0.444 Train-Acc:0.798
I:100 Test-Err:0.437 Test-Acc:0.8029 Train-Err:0.436 Train-Acc:0.805
I:110 Test-Err:0.431 Test-Acc:0.8024 Train-Err:0.420 Train-Acc:0.818
I:120 Test-Err:0.433 Test-Acc:0.8006 Train-Err:0.419 Train-Acc:0.824
I:130 Test-Err:0.432 Test-Acc:0.8003 Train-Err:0.424 Train-Acc:0.826
I:140 Test-Err:0.440 Test-Acc:0.7998 Train-Err:0.430 Train-Acc:0.832
I:150 Test-Err:0.437 Test-Acc:0.7967 Train-Err:0.410 Train-Acc:0.818
I:160 Test-Err:0.435 Test-Acc:0.8052 Train-Err:0.422 Train-Acc:0.811
I:170 Test-Err:0.436 Test-Acc:0.7953 Train-Err:0.412 Train-Acc:0.846
I:180 Test-Err:0.432 Test-Acc:0.8049 Train-Err:0.405 Train-Acc:0.842
I:190 Test-Err:0.430 Test-Acc:0.8013 Train-Err:0.404 Train-Acc:0.84
I:200 Test-Err:0.430 Test-Acc:0.8089 Train-Err:0.415 Train-Acc:0.844
I:210 Test-Err:0.427 Test-Acc:0.801 Train-Err:0.398 Train-Acc:0.844
I:220 Test-Err:0.426 Test-Acc:0.8064 Train-Err:0.392 Train-Acc:0.841
I:230 Test-Err:0.424 Test-Acc:0.8088 Train-Err:0.389 Train-Acc:0.849
I:240 Test-Err:0.425 Test-Acc:0.8114 Train-Err:0.392 Train-Acc:0.85
I:250 Test-Err:0.424 Test-Acc:0.8074 Train-Err:0.385 Train-Acc:0.851
I:260 Test-Err:0.423 Test-Acc:0.8129 Train-Err:0.390 Train-Acc:0.853
I:270 Test-Err:0.421 Test-Acc:0.8071 Train-Err:0.378 Train-Acc:0.866
I:280 Test-Err:0.421 Test-Acc:0.8124 Train-Err:0.393 Train-Acc:0.856
I:290 Test-Err:0.418 Test-Acc:0.806 Train-Err:0.375 Train-Acc:0.862
"""

