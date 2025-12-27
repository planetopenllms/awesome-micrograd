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


model = nn.Sequential( [nn.Linear(pixels_per_image, hidden_size), nn.ReLU(),
                        nn.Linear( hidden_size, num_labels)] )

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
I:0 Train-Err:0.72340107 Train-Acc:0.535   Test-Err:0.60074055 Test-Acc:0.6495
 I:10 Train-Err:0.11427351 Train-Acc:0.967   Test-Err:0.29972774 Test-Acc:0.856
 I:20 Train-Err:0.07181338 Train-Acc:0.977   Test-Err:0.30229405 Test-Acc:0.8505
 I:30 Train-Err:0.05436911 Train-Acc:0.979   Test-Err:0.31703812 Test-Acc:0.8401
 I:40 Train-Err:0.04437655 Train-Acc:0.979   Test-Err:0.33039916 Test-Acc:0.8317
 I:50 Train-Err:0.038131103 Train-Acc:0.981   Test-Err:0.33874923 Test-Acc:0.8269
 I:60 Train-Err:0.03379036 Train-Acc:0.981   Test-Err:0.34317666 Test-Acc:0.8239
 I:70 Train-Err:0.030651657 Train-Acc:0.981   Test-Err:0.34791315 Test-Acc:0.8213
 I:80 Train-Err:0.02825887 Train-Acc:0.981   Test-Err:0.35180902 Test-Acc:0.8192
 I:90 Train-Err:0.026346043 Train-Acc:0.982   Test-Err:0.35415533 Test-Acc:0.8209
 I:100 Train-Err:0.024900172 Train-Acc:0.983   Test-Err:0.355324 Test-Acc:0.8208
 I:110 Train-Err:0.023531215 Train-Acc:0.983   Test-Err:0.3567276 Test-Acc:0.8204
 I:120 Train-Err:0.022202648 Train-Acc:0.985   Test-Err:0.358954 Test-Acc:0.8207
 I:130 Train-Err:0.020818228 Train-Acc:0.986   Test-Err:0.36173594 Test-Acc:0.8214
 I:140 Train-Err:0.019814791 Train-Acc:0.986   Test-Err:0.363782 Test-Acc:0.8216
 I:150 Train-Err:0.01899713 Train-Acc:0.986   Test-Err:0.36595035 Test-Acc:0.8214
 I:160 Train-Err:0.018353658 Train-Acc:0.986   Test-Err:0.36767694 Test-Acc:0.8199
 I:170 Train-Err:0.01781888 Train-Acc:0.986   Test-Err:0.36967427 Test-Acc:0.8188
 I:180 Train-Err:0.01736207 Train-Acc:0.986   Test-Err:0.37147698 Test-Acc:0.8183
 I:190 Train-Err:0.016968243 Train-Acc:0.987   Test-Err:0.37287036 Test-Acc:0.8175
 I:200 Train-Err:0.016636686 Train-Acc:0.987   Test-Err:0.3740813 Test-Acc:0.8167
 I:210 Train-Err:0.016309723 Train-Acc:0.988   Test-Err:0.37515277 Test-Acc:0.8172
 I:220 Train-Err:0.016018825 Train-Acc:0.988   Test-Err:0.37612423 Test-Acc:0.816
 I:230 Train-Err:0.0157511 Train-Acc:0.988   Test-Err:0.37709323 Test-Acc:0.8145
 I:240 Train-Err:0.01550582 Train-Acc:0.988   Test-Err:0.378006 Test-Acc:0.8141
 I:250 Train-Err:0.015290265 Train-Acc:0.988   Test-Err:0.3788511 Test-Acc:0.813
 I:260 Train-Err:0.015083465 Train-Acc:0.988   Test-Err:0.3794964 Test-Acc:0.8117
 I:270 Train-Err:0.014900112 Train-Acc:0.988   Test-Err:0.3801948 Test-Acc:0.8112
 I:280 Train-Err:0.014735186 Train-Acc:0.988   Test-Err:0.38083836 Test-Acc:0.8105
 I:290 Train-Err:0.014586992 Train-Acc:0.989   Test-Err:0.38211706 Test-Acc:0.8101
 I:300 Train-Err:0.014448899 Train-Acc:0.989   Test-Err:0.3822238 Test-Acc:0.8104
 I:310 Train-Err:0.014327732 Train-Acc:0.989   Test-Err:0.38334867 Test-Acc:0.8102
 I:320 Train-Err:0.014215632 Train-Acc:0.989   Test-Err:0.38342798 Test-Acc:0.8094
 I:330 Train-Err:0.014114684 Train-Acc:0.989   Test-Err:0.38394699 Test-Acc:0.8087
 I:340 Train-Err:0.014017821 Train-Acc:0.989   Test-Err:0.38480493 Test-Acc:0.8074
 I:349 Train-Err:0.013937371 Train-Acc:0.989   Test-Err:0.38501376 Test-Acc:0.8067

 
 or

  I:0 Train-Err:0.74830335 Train-Acc:0.548   Test-Err:0.6158054 Test-Acc:0.6417
 I:10 Train-Err:0.13460419 Train-Acc:0.971   Test-Err:0.33397514 Test-Acc:0.8573
 I:20 Train-Err:0.08100569 Train-Acc:0.981   Test-Err:0.30984253 Test-Acc:0.8589
 I:30 Train-Err:0.058623902 Train-Acc:0.981   Test-Err:0.31425017 Test-Acc:0.8563
 I:40 Train-Err:0.046935774 Train-Acc:0.981   Test-Err:0.32129526 Test-Acc:0.8515
 I:50 Train-Err:0.040011715 Train-Acc:0.984   Test-Err:0.33514145 Test-Acc:0.8492
 I:60 Train-Err:0.03437703 Train-Acc:0.989   Test-Err:0.34176835 Test-Acc:0.8474
 I:70 Train-Err:0.030137824 Train-Acc:0.99   Test-Err:0.35218453 Test-Acc:0.8429
 I:80 Train-Err:0.026984293 Train-Acc:0.99   Test-Err:0.35865504 Test-Acc:0.8383
 I:90 Train-Err:0.024501793 Train-Acc:0.991   Test-Err:0.3616113 Test-Acc:0.836
 I:100 Train-Err:0.022719864 Train-Acc:0.991   Test-Err:0.36360598 Test-Acc:0.8336
 I:110 Train-Err:0.021315368 Train-Acc:0.991   Test-Err:0.36188877 Test-Acc:0.8324
 I:120 Train-Err:0.020252245 Train-Acc:0.991   Test-Err:0.3640202 Test-Acc:0.8319
 I:130 Train-Err:0.019311626 Train-Acc:0.991   Test-Err:0.36243054 Test-Acc:0.8297
 I:140 Train-Err:0.018559603 Train-Acc:0.991   Test-Err:0.36258665 Test-Acc:0.8278
 I:150 Train-Err:0.017910078 Train-Acc:0.991   Test-Err:0.36329755 Test-Acc:0.828
 I:160 Train-Err:0.017363552 Train-Acc:0.991   Test-Err:0.36543378 Test-Acc:0.8266
 I:170 Train-Err:0.016859159 Train-Acc:0.991   Test-Err:0.36545765 Test-Acc:0.8276
 I:180 Train-Err:0.016431646 Train-Acc:0.991   Test-Err:0.3676907 Test-Acc:0.8285
 I:190 Train-Err:0.016053608 Train-Acc:0.991   Test-Err:0.36930045 Test-Acc:0.8274
 I:200 Train-Err:0.015718985 Train-Acc:0.991   Test-Err:0.3702436 Test-Acc:0.8293
 I:210 Train-Err:0.015354824 Train-Acc:0.991   Test-Err:0.37125885 Test-Acc:0.827
 I:220 Train-Err:0.0151410615 Train-Acc:0.991   Test-Err:0.3727412 Test-Acc:0.8272
 I:230 Train-Err:0.01483373 Train-Acc:0.991   Test-Err:0.37461567 Test-Acc:0.8244
 I:240 Train-Err:0.014590886 Train-Acc:0.991   Test-Err:0.37574184 Test-Acc:0.8236
 I:250 Train-Err:0.014342365 Train-Acc:0.991   Test-Err:0.3778399 Test-Acc:0.8224
 I:260 Train-Err:0.01416446 Train-Acc:0.991   Test-Err:0.37818488 Test-Acc:0.8214
 I:270 Train-Err:0.013921858 Train-Acc:0.991   Test-Err:0.38006887 Test-Acc:0.821
 I:280 Train-Err:0.01449179 Train-Acc:0.991   Test-Err:0.3831643 Test-Acc:0.816
 I:290 Train-Err:0.013561413 Train-Acc:0.991   Test-Err:0.38235238 Test-Acc:0.8197
 I:300 Train-Err:0.013324749 Train-Acc:0.991   Test-Err:0.38340795 Test-Acc:0.8183
 I:310 Train-Err:0.013320643 Train-Acc:0.991   Test-Err:0.38422823 Test-Acc:0.8167
 I:320 Train-Err:0.013271884 Train-Acc:0.991   Test-Err:0.38580078 Test-Acc:0.8184
 I:330 Train-Err:0.013205145 Train-Acc:0.991   Test-Err:0.3854479 Test-Acc:0.8167
 I:340 Train-Err:0.012913284 Train-Acc:0.991   Test-Err:0.38558492 Test-Acc:0.8176
 I:349 Train-Err:0.012612738 Train-Acc:0.991   Test-Err:0.38849232 Test-Acc:0.8192
 """

