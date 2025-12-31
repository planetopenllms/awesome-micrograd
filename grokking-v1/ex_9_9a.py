##########
#  mintorch version of ex_9_9.py
#     "old style" - uses MSE loss  (instead of softmax and cross-entropy)

from mintorch import Tensor, nn, optim

import sys, numpy as np
import mnist

(images, labels), (test_images, test_labels) = mnist.load_data()


batch_size = 100
lr = 0.001
iterations = 300
hidden_size = 100
pixels_per_image = 784 
num_labels = 10


model = nn.Sequential( nn.Linear(pixels_per_image, hidden_size, bias=False), 
                       nn.Tanh(),
                       nn.Dropout(p=0.5),
                       nn.Linear( hidden_size, num_labels, bias=False) )
## note - use same weight init as in orginal sample!!!
##    and random seed
np.random.seed(1)
model[0].weight.data = 0.02*np.random.random((pixels_per_image,hidden_size))-0.01
model[3].weight.data = 0.2*np.random.random((hidden_size,num_labels)) - 0.1



criterion = nn.MSELoss()
optimizer = optim.SGD(parameters=model.parameters(), lr=lr)



for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    model.train()
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i*batch_size),((i+1)*batch_size))
        # predict
        x = Tensor(images[batch_start:batch_end])
        y_hat = model( x )
        # compare   
        y = Tensor(labels[batch_start:batch_end])
        loss   = criterion(y_hat, y)
        ##   note - add sum() here to add-up/sum the multi-category loss
        error += loss.data.sum()    
        for k in range(batch_size):
            correct_cnt += int(np.argmax(y_hat.data[k:k+1]) == 
                                 np.argmax(labels[batch_start+k:batch_start+k+1])) 
        ## Learn
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    test_error, test_correct_cnt = (0.0,0)
    model.eval()
    for i in range(len(test_images)):
        x_test = Tensor(test_images[i:i+1])
        y_hat  = model( x_test )
    
        y_test = Tensor(test_labels[i:i+1]) 
        loss = criterion(y_hat, y_test)
        test_error += loss.data.sum()
        test_correct_cnt += int(np.argmax(y_hat.data) == 
                                np.argmax(test_labels[i:i+1]))
    if(j % 10 == 0):
        sys.stdout.write("\n"+ 
         f"I:{j}" + 
         f" Test-Err: {test_error/len(test_images)}" +
         f" Test-Acc:{test_correct_cnt/len(test_images)}" +
         f" Train-Err: {error/len(images)}" +
         f" Train-Acc:{correct_cnt/len(images)}")


print("\nbye")


"""
I:0 Test-Err: 1.3857321739196777 Test-Acc:0.2482 Train-Err: 1.6682491302490234 Train-Acc:0.199
I:10 Test-Err: 0.5172490477561951 Test-Acc:0.7561 Train-Err: 0.4591355621814728 Train-Acc:0.836
I:20 Test-Err: 0.48684751987457275 Test-Acc:0.7958 Train-Err: 0.4235416054725647 Train-Acc:0.866
I:30 Test-Err: 0.4900711476802826 Test-Acc:0.7843 Train-Err: 0.411918580532074 Train-Acc:0.88
I:40 Test-Err: 0.47975003719329834 Test-Acc:0.7989 Train-Err: 0.3985011577606201 Train-Acc:0.894
I:50 Test-Err: 0.48293760418891907 Test-Acc:0.7893 Train-Err: 0.3843042254447937 Train-Acc:0.901
I:60 Test-Err: 0.47739842534065247 Test-Acc:0.7948 Train-Err: 0.3843301832675934 Train-Acc:0.908
I:70 Test-Err: 0.494277685880661 Test-Acc:0.7916 Train-Err: 0.37548503279685974 Train-Acc:0.91
I:80 Test-Err: 0.5282474160194397 Test-Acc:0.7593 Train-Err: 0.3744301199913025 Train-Acc:0.925
I:90 Test-Err: 0.48239466547966003 Test-Acc:0.792 Train-Err: 0.3715146481990814 Train-Acc:0.929
I:100 Test-Err: 0.4952335059642792 Test-Acc:0.7898 Train-Err: 0.36510348320007324 Train-Acc:0.923
I:110 Test-Err: 0.5167689323425293 Test-Acc:0.7912 Train-Err: 0.35381218791007996 Train-Acc:0.929
I:120 Test-Err: 0.5126932263374329 Test-Acc:0.775 Train-Err: 0.3513677716255188 Train-Acc:0.945
I:130 Test-Err: 0.49472296237945557 Test-Acc:0.7885 Train-Err: 0.3512033522129059 Train-Acc:0.934
I:140 Test-Err: 0.5065956115722656 Test-Acc:0.7763 Train-Err: 0.3455176055431366 Train-Acc:0.944
I:150 Test-Err: 0.5044281482696533 Test-Acc:0.7863 Train-Err: 0.3436606228351593 Train-Acc:0.944
I:160 Test-Err: 0.5100922584533691 Test-Acc:0.78 Train-Err: 0.3399089276790619 Train-Acc:0.945
I:170 Test-Err: 0.5053436756134033 Test-Acc:0.7785 Train-Err: 0.34457722306251526 Train-Acc:0.945
I:180 Test-Err: 0.5389517545700073 Test-Acc:0.7575 Train-Err: 0.34337517619132996 Train-Acc:0.949
I:190 Test-Err: 0.5112658739089966 Test-Acc:0.7818 Train-Err: 0.32849380373954773 Train-Acc:0.96
I:200 Test-Err: 0.5223638415336609 Test-Acc:0.7855 Train-Err: 0.3326694667339325 Train-Acc:0.951
I:210 Test-Err: 0.5180033445358276 Test-Acc:0.7734 Train-Err: 0.32966503500938416 Train-Acc:0.952
I:220 Test-Err: 0.5293012857437134 Test-Acc:0.7643 Train-Err: 0.33573511242866516 Train-Acc:0.95
I:230 Test-Err: 0.5263123512268066 Test-Acc:0.7663 Train-Err: 0.3283083438873291 Train-Acc:0.955
I:240 Test-Err: 0.5308598279953003 Test-Acc:0.7662 Train-Err: 0.32792019844055176 Train-Acc:0.961
I:250 Test-Err: 0.5552543997764587 Test-Acc:0.759 Train-Err: 0.3252304494380951 Train-Acc:0.963
I:260 Test-Err: 0.5295979976654053 Test-Acc:0.7689 Train-Err: 0.3262745440006256 Train-Acc:0.958
I:270 Test-Err: 0.5349365472793579 Test-Acc:0.7643 Train-Err: 0.32149994373321533 Train-Acc:0.96
I:280 Test-Err: 0.5497874617576599 Test-Acc:0.7555 Train-Err: 0.33185911178588867 Train-Acc:0.956
I:290 Test-Err: 0.5319324731826782 Test-Acc:0.767 Train-Err: 0.31572869420051575 Train-Acc:0.953
"""