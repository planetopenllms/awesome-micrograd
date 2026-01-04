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
#
#   ----
##   checking in with chatgpt 
#          the recommendation is to use (NEW!) layer1_pre for reluderiv
#          and use the dropout mask =* 2 for both forward and backward   


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


## drooput params
dropout_p = 0.5
keep_prob = 1.0 - dropout_p   # 0.5
scale = 1.0 / keep_prob       # 2


for j in range(iterations):
    error, correct_cnt = (0.0,0)
    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1_pre = np.dot(layer_0, weights_0_1)
        layer_1 = relu(layer_1_pre)
        # inverted dropout
        dropout_mask = (np.random.rand(*layer_1.shape) < keep_prob) * scale
        layer_1 *= dropout_mask

        layer_2 = np.dot(layer_1,weights_1_2)

        ## backward
        error += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))
        layer_2_delta = (labels[i:i+1] - layer_2)
        
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)
        layer_1_delta *= relu2deriv(layer_1_pre)   ## note: uses layer_1_pre!!!
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
I:0 Test-Err:0.610 Test-Acc:0.6638 Train-Err:0.853 Train-Acc:0.438
I:10 Test-Err:0.334 Test-Acc:0.8461 Train-Err:0.348 Train-Acc:0.871
I:20 Test-Err:0.295 Test-Acc:0.8692 Train-Err:0.289 Train-Acc:0.906
I:30 Test-Err:0.294 Test-Acc:0.8794 Train-Err:0.245 Train-Acc:0.931
I:40 Test-Err:0.280 Test-Acc:0.8759 Train-Err:0.236 Train-Acc:0.944
I:50 Test-Err:0.279 Test-Acc:0.8778 Train-Err:0.217 Train-Acc:0.946
I:60 Test-Err:0.283 Test-Acc:0.8778 Train-Err:0.202 Train-Acc:0.964
I:70 Test-Err:0.281 Test-Acc:0.8763 Train-Err:0.187 Train-Acc:0.975
I:80 Test-Err:0.280 Test-Acc:0.8756 Train-Err:0.192 Train-Acc:0.968
I:90 Test-Err:0.297 Test-Acc:0.8757 Train-Err:0.182 Train-Acc:0.973
I:100 Test-Err:0.279 Test-Acc:0.8773 Train-Err:0.176 Train-Acc:0.969
I:110 Test-Err:0.283 Test-Acc:0.8768 Train-Err:0.176 Train-Acc:0.97
I:120 Test-Err:0.284 Test-Acc:0.8758 Train-Err:0.179 Train-Acc:0.975
I:130 Test-Err:0.289 Test-Acc:0.874 Train-Err:0.169 Train-Acc:0.976
I:140 Test-Err:0.285 Test-Acc:0.8751 Train-Err:0.168 Train-Acc:0.978
I:150 Test-Err:0.285 Test-Acc:0.8763 Train-Err:0.161 Train-Acc:0.984
I:160 Test-Err:0.284 Test-Acc:0.8759 Train-Err:0.175 Train-Acc:0.974
I:170 Test-Err:0.296 Test-Acc:0.8755 Train-Err:0.155 Train-Acc:0.977
I:180 Test-Err:0.291 Test-Acc:0.8741 Train-Err:0.173 Train-Acc:0.976
I:190 Test-Err:0.283 Test-Acc:0.8754 Train-Err:0.160 Train-Acc:0.985
I:200 Test-Err:0.289 Test-Acc:0.8752 Train-Err:0.165 Train-Acc:0.982
I:210 Test-Err:0.289 Test-Acc:0.8758 Train-Err:0.159 Train-Acc:0.983
I:220 Test-Err:0.292 Test-Acc:0.8752 Train-Err:0.155 Train-Acc:0.988
I:230 Test-Err:0.293 Test-Acc:0.8719 Train-Err:0.154 Train-Acc:0.984
I:240 Test-Err:0.281 Test-Acc:0.8743 Train-Err:0.148 Train-Acc:0.983
I:250 Test-Err:0.291 Test-Acc:0.8711 Train-Err:0.156 Train-Acc:0.984
I:260 Test-Err:0.286 Test-Acc:0.873 Train-Err:0.162 Train-Acc:0.981
I:270 Test-Err:0.283 Test-Acc:0.8735 Train-Err:0.144 Train-Acc:0.99
I:280 Test-Err:0.293 Test-Acc:0.8686 Train-Err:0.149 Train-Acc:0.983
I:290 Test-Err:0.288 Test-Acc:0.8699 Train-Err:0.155 Train-Acc:0.981
"""
