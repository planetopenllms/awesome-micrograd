##########
#  mintorch version of ex_9_9.py
#     adds cross_entropy with integer labels (NOT one-hot encoded)

from mintorch import Tensor, nn, optim

import sys, numpy as np
import mnist

(images, labels), (test_images, test_labels) = mnist.load_data_v2()


lr = 0.02
iterations = 300
hidden_size = 100
pixels_per_image = 784 
num_labels = 10
batch_size = 100


model = nn.Sequential( nn.Linear(pixels_per_image, hidden_size, bias=False), 
                       nn.Tanh(),
                       nn.Dropout(p=0.5),
                       nn.Linear( hidden_size, num_labels, bias=False) )
## note - use same weight init as in orginal sample!!!
##    and random seed
np.random.seed(1)
model[0].weight.data = 0.02*np.random.random((pixels_per_image,hidden_size))-0.01
model[3].weight.data = 0.2*np.random.random((hidden_size,num_labels)) - 0.1



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(parameters=model.parameters(), lr=lr)



for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    model.train()
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end=((i * batch_size),((i+1)*batch_size))
        # predict
        x = Tensor(images[batch_start:batch_end])
        y_hat = model( x )
        # compare   
        y = Tensor(labels[batch_start:batch_end])
        loss   = criterion(y_hat, y)
        ##   note - add sum() here to add-up/sum the multi-category loss
        error += loss.data    
        for k in range(batch_size):
            correct_cnt += int(np.argmax(y_hat.data[k:k+1]) == 
                                 np.argmax(labels[batch_start+k:batch_start+k+1])) 
        ## Learn
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print( f"I:{j}, error:{error}, correct:{correct_cnt}")

    test_correct_cnt = 0
    model.eval()
    """
    for i in range(len(test_images)):
        layer_0 = test_images[i:i+1]
        layer_1 = tanh(np.dot(layer_0,weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))
    if(j % 10 == 0):
        sys.stdout.write("\n"+ 
         f"I:{j}" + 
         f" Test-Acc:{test_correct_cnt/len(test_images)}"+
         f" Train-Acc:{correct_cnt/len(images)}")
    """

print("\nbye")


"""
mintorch\autograd.py:299: RuntimeWarning: divide by zero encountered in log
  loss = -(np.log(softmax_output) * (target_dist)).sum(axis=-1).mean()
mintorch\autograd.py:299: RuntimeWarning: invalid value encountered in multiply
  loss = -(np.log(softmax_output) * (target_dist)).sum(axis=-1).mean()
I:0, error:nan, correct:201
I:1, error:nan, correct:90
I:2, error:nan, correct:96
I:3, error:nan, correct:102
I:4, error:nan, correct:105
I:5, error:nan, correct:96
I:6, error:126.77020263671875, correct:104
I:7, error:nan, correct:89
I:8, error:nan, correct:100
I:9, error:nan, correct:105
I:10, error:nan, correct:89
I:11, error:nan, correct:101
I:12, error:54.37588882446289, correct:97
I:13, error:54.608863830566406, correct:100
"""