##########
#  mintorch version of ex_8_2.py

from mintorch import Tensor, nn, optim

import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, y_train[0:1000])

one_hot_labels = np.zeros((len(labels),10))
for i,l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test),28*28) / 255
test_labels = np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
    test_labels[i][l] = 1


lr = 0.005
iterations = 350 
hidden_size = 40
pixels_per_image = 784  # 28x28px 
num_labels = 10


layer1 = nn.Linear(pixels_per_image, hidden_size)
layer2 = nn.Linear( hidden_size, num_labels)

## note - use same weight init as in orginal sample!!!
layer1.weight.data = 0.2*np.random.random((pixels_per_image,hidden_size)) - 0.1
layer2.weight.data = 0.2*np.random.random((hidden_size,num_labels)) - 0.1

model = nn.Sequential( [layer1, nn.ReLU(),
                        layer2] )

criterion = nn.MSELoss()
optimizer = optim.SGD(parameters=model.parameters(), lr=lr)



for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    
    for i in range(len(images)):
        ## predict
        x      = Tensor(images[i:i+1])
        y_hat =  model( x )
        ## compare
        y      = Tensor(labels[i:i+1])   
        ## --  error += np.sum((labels[i:i+1] - layer_2) ** 2)
        ##   note - MSELoss only sums up batch dimension (dim=0)
        ##              NOT the multi-category loss if present
        loss   = criterion(y_hat, y)
        ##   note - add sum() here to add-up/sum the multi-category loss
        error += loss.data.sum()    ## use loss.item() ???
        correct_cnt += int(np.argmax(y_hat.data) == \
                               np.argmax(labels[i:i+1])) 
        ## Learn
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    sys.stdout.write("\r I:"+str(j)+ \
                     " Train-Err:" + str(error/len(images)) +\
                     " Train-Acc:" + str(correct_cnt/len(images)) +\
                     "  ")


    if(j % 10 == 0 or j == iterations-1):
        error, correct_cnt = (0.0, 0)

        for i in range(len(test_images)):
            x_test = Tensor(test_images[i:i+1])
            y_hat  = model( x_test )
            
            y_test = Tensor(test_labels[i:i+1]) 
            loss = criterion(y_hat, y_test)
            error += loss.data.sum()
            correct_cnt += int(np.argmax(y_hat.data) == \
                                np.argmax(test_labels[i:i+1]))
        sys.stdout.write(" Test-Err:" + str(error/len(test_images)) +\
                         " Test-Acc:" + str(correct_cnt/len(test_images)) +\
                         "         " )
        print()


print("\nbye")

"""
 I:0 Train-Err:0.6437854 Train-Acc:0.592   Test-Err:0.5196275 Test-Acc:0.6764
 I:10 Train-Err:0.123614736 Train-Acc:0.973   Test-Err:0.28411096 Test-Acc:0.8677
 I:20 Train-Err:0.06862172 Train-Acc:0.994   Test-Err:0.2832052 Test-Acc:0.8666
 I:30 Train-Err:0.04301278 Train-Acc:0.998   Test-Err:0.2903944 Test-Acc:0.8614
 I:40 Train-Err:0.02852836 Train-Acc:0.999   Test-Err:0.2957293 Test-Acc:0.8565
 I:50 Train-Err:0.020082243 Train-Acc:0.999   Test-Err:0.29924402 Test-Acc:0.8529
 I:60 Train-Err:0.014799728 Train-Acc:0.999   Test-Err:0.3024055 Test-Acc:0.8497
 I:70 Train-Err:0.011213454 Train-Acc:0.999   Test-Err:0.30363646 Test-Acc:0.8478
 I:80 Train-Err:0.008763285 Train-Acc:0.999   Test-Err:0.30392343 Test-Acc:0.8486
 I:90 Train-Err:0.0070623825 Train-Acc:0.999   Test-Err:0.30594626 Test-Acc:0.8449
 I:100 Train-Err:0.005788165 Train-Acc:0.999   Test-Err:0.30709955 Test-Acc:0.8442
 I:110 Train-Err:0.004795009 Train-Acc:0.999   Test-Err:0.3085815 Test-Acc:0.8432
 I:120 Train-Err:0.003982355 Train-Acc:1.0   Test-Err:0.3102795 Test-Acc:0.8433
 I:130 Train-Err:0.0033131812 Train-Acc:1.0   Test-Err:0.31149244 Test-Acc:0.8409
 I:140 Train-Err:0.0027536075 Train-Acc:1.0   Test-Err:0.3124654 Test-Acc:0.8397
 I:150 Train-Err:0.0023030408 Train-Acc:1.0   Test-Err:0.3134914 Test-Acc:0.8392
 I:160 Train-Err:0.0019244569 Train-Acc:1.0   Test-Err:0.31451443 Test-Acc:0.8382
 I:170 Train-Err:0.0016129905 Train-Acc:1.0   Test-Err:0.31543374 Test-Acc:0.8362
 I:180 Train-Err:0.0013583514 Train-Acc:1.0   Test-Err:0.3162579 Test-Acc:0.8355
 I:190 Train-Err:0.0011521218 Train-Acc:1.0   Test-Err:0.31699052 Test-Acc:0.8349
 I:200 Train-Err:0.00097604643 Train-Acc:1.0   Test-Err:0.3176635 Test-Acc:0.8333
 I:210 Train-Err:0.0008321175 Train-Acc:1.0   Test-Err:0.3182687 Test-Acc:0.8329
 I:220 Train-Err:0.00071147265 Train-Acc:1.0   Test-Err:0.3187628 Test-Acc:0.8318
 I:230 Train-Err:0.0006104857 Train-Acc:1.0   Test-Err:0.31899792 Test-Acc:0.8322
 I:240 Train-Err:0.000530664 Train-Acc:1.0   Test-Err:0.31935847 Test-Acc:0.8318
 I:250 Train-Err:0.000463174 Train-Acc:1.0   Test-Err:0.3195723 Test-Acc:0.8318
 I:260 Train-Err:0.00041201708 Train-Acc:1.0   Test-Err:0.3197897 Test-Acc:0.8315
 I:270 Train-Err:0.00036382306 Train-Acc:1.0   Test-Err:0.3200498 Test-Acc:0.8313
 I:280 Train-Err:0.00032761318 Train-Acc:1.0   Test-Err:0.3202568 Test-Acc:0.8312
 I:290 Train-Err:0.00029652243 Train-Acc:1.0   Test-Err:0.3205419 Test-Acc:0.8309
 I:300 Train-Err:0.000268508 Train-Acc:1.0   Test-Err:0.32069498 Test-Acc:0.8298
 I:310 Train-Err:0.00024386332 Train-Acc:1.0   Test-Err:0.32094815 Test-Acc:0.8291
 I:320 Train-Err:0.00022524275 Train-Acc:1.0   Test-Err:0.32123294 Test-Acc:0.8284
 I:330 Train-Err:0.00020556158 Train-Acc:1.0   Test-Err:0.32151625 Test-Acc:0.8282
 I:340 Train-Err:0.00018873175 Train-Acc:1.0   Test-Err:0.3217621 Test-Acc:0.8281
 I:349 Train-Err:0.00017626908 Train-Acc:1.0   Test-Err:0.32199427 Test-Acc:0.8275
"""

