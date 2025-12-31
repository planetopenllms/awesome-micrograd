##########
#  mintorch version of ex_8_11.py

from mintorch import Tensor, nn, optim


import sys, numpy as np
import mnist

(images, labels), (test_images, test_labels) = mnist.load_data()


batch_size = 100
lr         = 0.001       
iterations = 300
pixels_per_image = 784
num_labels = 10
hidden_size = 100


model = nn.Sequential( nn.Linear(pixels_per_image, hidden_size, bias=False), 
                       nn.ReLU(),
                       nn.Dropout(p=0.5),
                       nn.Linear( hidden_size, num_labels, bias=False) )
## note - use same weight init as in orginal sample!!!
##    and random seed
np.random.seed(1)
model[0].weight.data = 0.2*np.random.random((pixels_per_image,hidden_size)) - 0.1
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
        error += loss.data  
        for k in range(batch_size):
            correct_cnt += int(np.argmax(y_hat.data[k:k+1]) == 
                                 np.argmax(labels[batch_start+k:batch_start+k+1])) 
        ## Learn
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if(j%10 == 0):
        test_error, test_correct_cnt = (0.0, 0)

        model.eval()
        for i in range(len(test_images)):
            x_test = Tensor(test_images[i:i+1])
            y_hat  = model( x_test )
    
            y_test = Tensor(test_labels[i:i+1]) 
            loss = criterion(y_hat, y_test)
            test_error += loss.data 
            test_correct_cnt += int(np.argmax(y_hat.data) == 
                                np.argmax(test_labels[i:i+1]))

        sys.stdout.write("\n" + 
                         f"I:{j}" + 
                         f" Test-Err:{test_error/len(test_images)}" + 
                         f" Test-Acc:{test_correct_cnt/len(test_images)}" + 
                         f" Train-Err:{error/len(images)}" + 
                         f" Train-Acc:{correct_cnt/len(images)}")

print( "\nbye")

"""
I:0 Test-Err:0.7480621933937073 Test-Acc:0.4945 Train-Err:1.1123535633087158 Train-Acc:0.281
I:10 Test-Err:0.4111851453781128 Test-Acc:0.8077 Train-Err:0.43894103169441223 Train-Acc:0.8
I:20 Test-Err:0.3371354639530182 Test-Acc:0.8467 Train-Err:0.35669276118278503 Train-Acc:0.86
I:30 Test-Err:0.3160116672515869 Test-Acc:0.8675 Train-Err:0.30979764461517334 Train-Acc:0.881
I:40 Test-Err:0.29532650113105774 Test-Acc:0.8749 Train-Err:0.29402318596839905 Train-Acc:0.916
I:50 Test-Err:0.29241254925727844 Test-Acc:0.8769 Train-Err:0.2691691219806671 Train-Acc:0.913
I:60 Test-Err:0.28700530529022217 Test-Acc:0.8773 Train-Err:0.25245627760887146 Train-Acc:0.938
I:70 Test-Err:0.283626526594162 Test-Acc:0.8801 Train-Err:0.23454555869102478 Train-Acc:0.944
I:80 Test-Err:0.2796790897846222 Test-Acc:0.8804 Train-Err:0.23119592666625977 Train-Acc:0.938
I:90 Test-Err:0.2932603061199188 Test-Acc:0.883 Train-Err:0.22994393110275269 Train-Acc:0.949
I:100 Test-Err:0.27299341559410095 Test-Acc:0.8842 Train-Err:0.2145334929227829 Train-Acc:0.949
I:110 Test-Err:0.2736591696739197 Test-Acc:0.8846 Train-Err:0.2050342708826065 Train-Acc:0.958
I:120 Test-Err:0.2778584361076355 Test-Acc:0.8839 Train-Err:0.20457546412944794 Train-Acc:0.96
I:130 Test-Err:0.2750869393348694 Test-Acc:0.8801 Train-Err:0.1957615613937378 Train-Acc:0.964
I:140 Test-Err:0.2751281261444092 Test-Acc:0.8818 Train-Err:0.19373349845409393 Train-Acc:0.97
I:150 Test-Err:0.27727359533309937 Test-Acc:0.8784 Train-Err:0.1895894855260849 Train-Acc:0.97
I:160 Test-Err:0.27852576971054077 Test-Acc:0.8824 Train-Err:0.20046502351760864 Train-Acc:0.966
I:170 Test-Err:0.27713853120803833 Test-Acc:0.8831 Train-Err:0.17903585731983185 Train-Acc:0.966
I:180 Test-Err:0.2812690734863281 Test-Acc:0.8799 Train-Err:0.1861748993396759 Train-Acc:0.969
I:190 Test-Err:0.27409690618515015 Test-Acc:0.88 Train-Err:0.1789511740207672 Train-Acc:0.982
I:200 Test-Err:0.28144875168800354 Test-Acc:0.8784 Train-Err:0.18476909399032593 Train-Acc:0.966
I:210 Test-Err:0.2843409478664398 Test-Acc:0.8787 Train-Err:0.18025454878807068 Train-Acc:0.978
I:220 Test-Err:0.28519225120544434 Test-Acc:0.8782 Train-Err:0.1730218082666397 Train-Acc:0.978
I:230 Test-Err:0.2793351411819458 Test-Acc:0.8817 Train-Err:0.16894347965717316 Train-Acc:0.979
I:240 Test-Err:0.27898260951042175 Test-Acc:0.8787 Train-Err:0.17076027393341064 Train-Acc:0.976
I:250 Test-Err:0.28172722458839417 Test-Acc:0.8771 Train-Err:0.17674162983894348 Train-Acc:0.977
I:260 Test-Err:0.2746571898460388 Test-Acc:0.882 Train-Err:0.1833568513393402 Train-Acc:0.968
I:270 Test-Err:0.27977538108825684 Test-Acc:0.879 Train-Err:0.16650770604610443 Train-Acc:0.978
I:280 Test-Err:0.28211578726768494 Test-Acc:0.8761 Train-Err:0.16105206310749054 Train-Acc:0.98
I:290 Test-Err:0.27884843945503235 Test-Acc:0.8792 Train-Err:0.17355069518089294 Train-Acc:0.972
"""

