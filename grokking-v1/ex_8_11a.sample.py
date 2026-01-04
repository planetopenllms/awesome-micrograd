##
## q: can you rewrite to a version that defines the loss as a "mean"?
##
##  Below is a clean rewrite of your code where the loss is explicitly 
#   defined as the mean over the batch, 
#   and the gradients follow naturally from that definition.
#
##  ...
##  Practical takeaway (this matters for learning rate)
#   Loss definition  | 	Gradient scale
#   mean over batch  |	 1 / B
#   mean over all elements (PyTorch)	| 1 / (B * C)    B=batch, C=(output classes)
#   sum	             |  no normalization


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
        batch_start = i * batch_size
        batch_end   = (i + 1) * batch_size

        # Forward pass
        layer_0 = images[batch_start:batch_end]                 # (B, D)
        layer_1 = relu(np.dot(layer_0, weights_0_1))            # (B, H)
        layer_2 = np.dot(layer_1, weights_1_2)                  # (B, C)

        # ----- Mean loss -----
        batch_error = np.mean((labels[batch_start:batch_end] - layer_2) ** 2)
        error += batch_error

        for k in range(batch_size):
            correct_cnt += int(
                np.argmax(layer_2[k:k+1]) ==
                np.argmax(labels[batch_start+k:batch_start+k+1])
            )

        # ----- Backprop (mean loss) -----
        # d/dy (1/B * sum (y_hat - y)^2) = 2/B * (y_hat - y)
        ##  change to 
        ##     loss = 0.5 * mean((y - y_hat)^2)   (cuts out *2)
        layer_2_delta = (layer_2 - labels[batch_start:batch_end])

        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        # ----- Mean gradients -----
        weights_1_2 -= alpha * (layer_1.T.dot(layer_2_delta) / batch_size)
        weights_0_1 -= alpha * (layer_0.T.dot(layer_1_delta) / batch_size)

            
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
I:0 Test-Err:0.780 Test-Acc:0.474 Train-Err:0.000 Train-Acc:0.304
I:10 Test-Err:0.501 Test-Acc:0.7807 Train-Err:0.000 Train-Acc:0.846
I:20 Test-Err:0.452 Test-Acc:0.8097 Train-Err:0.000 Train-Acc:0.885
I:30 Test-Err:0.427 Test-Acc:0.8229 Train-Err:0.000 Train-Acc:0.904
I:40 Test-Err:0.412 Test-Acc:0.8319 Train-Err:0.000 Train-Acc:0.921
I:50 Test-Err:0.403 Test-Acc:0.8374 Train-Err:0.000 Train-Acc:0.928
I:60 Test-Err:0.397 Test-Acc:0.8413 Train-Err:0.000 Train-Acc:0.941
I:70 Test-Err:0.394 Test-Acc:0.8425 Train-Err:0.000 Train-Acc:0.944
I:80 Test-Err:0.392 Test-Acc:0.8421 Train-Err:0.000 Train-Acc:0.954
I:90 Test-Err:0.392 Test-Acc:0.8424 Train-Err:0.000 Train-Acc:0.959
I:100 Test-Err:0.393 Test-Acc:0.8423 Train-Err:0.000 Train-Acc:0.962
I:110 Test-Err:0.394 Test-Acc:0.8411 Train-Err:0.000 Train-Acc:0.966
I:120 Test-Err:0.395 Test-Acc:0.8396 Train-Err:0.000 Train-Acc:0.967
I:130 Test-Err:0.397 Test-Acc:0.8386 Train-Err:0.000 Train-Acc:0.971
I:140 Test-Err:0.399 Test-Acc:0.8375 Train-Err:0.000 Train-Acc:0.973
I:150 Test-Err:0.401 Test-Acc:0.8359 Train-Err:0.000 Train-Acc:0.977
I:160 Test-Err:0.403 Test-Acc:0.835 Train-Err:0.000 Train-Acc:0.981
I:170 Test-Err:0.405 Test-Acc:0.835 Train-Err:0.000 Train-Acc:0.983
I:180 Test-Err:0.409 Test-Acc:0.8322 Train-Err:0.000 Train-Acc:0.983
I:190 Test-Err:0.426 Test-Acc:0.8181 Train-Err:0.000 Train-Acc:0.98
I:200 Test-Err:0.410 Test-Acc:0.8325 Train-Err:0.000 Train-Acc:0.986
I:210 Test-Err:0.415 Test-Acc:0.8285 Train-Err:0.000 Train-Acc:0.985
I:220 Test-Err:0.426 Test-Acc:0.8185 Train-Err:0.000 Train-Acc:0.984
I:230 Test-Err:0.416 Test-Acc:0.8304 Train-Err:0.000 Train-Acc:0.987
I:240 Test-Err:0.427 Test-Acc:0.82 Train-Err:0.000 Train-Acc:0.986
I:250 Test-Err:0.419 Test-Acc:0.8296 Train-Err:0.000 Train-Acc:0.988
I:260 Test-Err:0.422 Test-Acc:0.8281 Train-Err:0.000 Train-Acc:0.989
I:270 Test-Err:0.427 Test-Acc:0.8267 Train-Err:0.000 Train-Acc:0.989
I:280 Test-Err:0.430 Test-Acc:0.8249 Train-Err:0.000 Train-Acc:0.989
I:290 Test-Err:0.433 Test-Acc:0.8237 Train-Err:0.000 Train-Acc:0.985
"""