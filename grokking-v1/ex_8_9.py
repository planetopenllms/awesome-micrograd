##########
#  mintorch version of ex_8_9.py

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
iterations = 300
hidden_size = 100
pixels_per_image = 784   # 28x28px
num_labels = 10


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
    error, correct_cnt = (0.0,0)
    
    model.train()
    for i in range(len(images)):
        # predict
        x = Tensor(images[i:i+1])
        y_hat = model(x)
        ## compare
        y = Tensor(labels[i:i+1])
        loss   = criterion(y_hat, y)
        ##   note - add sum() here to add-up/sum the multi-category loss
        error += loss.data.sum()    ## use loss.item() ???
        correct_cnt += int(np.argmax(y_hat.data) == 
                               np.argmax(labels[i:i+1])) 
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
            test_error += loss.data.sum()
            test_correct_cnt += int(np.argmax(y_hat.data) == 
                                np.argmax(test_labels[i:i+1]))
  
        sys.stdout.write("\n" + 
                         f"I:{j}" + 
                         f" Test-Err:{test_error/len(test_images)}" + 
                         f" Test-Acc:{test_correct_cnt/len(test_images)}" + 
                         f" Train-Err:{error/len(images)}" + 
                         f" Train-Acc:{correct_cnt/len(images)}")
        
print("\nbye")


"""
I:0 Test-Err:0.5163827538490295 Test-Acc:0.6986 Train-Err:0.7577823400497437 Train-Acc:0.514
I:10 Test-Err:0.31066447496414185 Test-Acc:0.8585 Train-Err:0.3164075016975403 Train-Acc:0.899
I:20 Test-Err:0.2907479405403137 Test-Acc:0.8731 Train-Err:0.27134883403778076 Train-Acc:0.922
I:30 Test-Err:0.29635968804359436 Test-Acc:0.8755 Train-Err:0.23357102274894714 Train-Acc:0.945
I:40 Test-Err:0.2840403914451599 Test-Acc:0.8714 Train-Err:0.23336896300315857 Train-Acc:0.957
I:50 Test-Err:0.2798762321472168 Test-Acc:0.8714 Train-Err:0.20221692323684692 Train-Acc:0.963
I:60 Test-Err:0.2895442247390747 Test-Acc:0.8716 Train-Err:0.19419768452644348 Train-Acc:0.971
I:70 Test-Err:0.2842075824737549 Test-Acc:0.8705 Train-Err:0.18990235030651093 Train-Acc:0.976
I:80 Test-Err:0.30322733521461487 Test-Acc:0.8647 Train-Err:0.1914406269788742 Train-Acc:0.979
I:90 Test-Err:0.30241847038269043 Test-Acc:0.8708 Train-Err:0.17715196311473846 Train-Acc:0.981
I:100 Test-Err:0.2881709635257721 Test-Acc:0.873 Train-Err:0.1736833155155182 Train-Acc:0.977
I:110 Test-Err:0.28254032135009766 Test-Acc:0.8767 Train-Err:0.1689637154340744 Train-Acc:0.976
I:120 Test-Err:0.2894812226295471 Test-Acc:0.8669 Train-Err:0.17494790256023407 Train-Acc:0.98
I:130 Test-Err:0.29905182123184204 Test-Acc:0.86 Train-Err:0.17317166924476624 Train-Acc:0.978
I:140 Test-Err:0.2960337996482849 Test-Acc:0.8691 Train-Err:0.17064881324768066 Train-Acc:0.983
I:150 Test-Err:0.29952049255371094 Test-Acc:0.8671 Train-Err:0.1663658320903778 Train-Acc:0.981
I:160 Test-Err:0.30065062642097473 Test-Acc:0.8656 Train-Err:0.1762567013502121 Train-Acc:0.975
I:170 Test-Err:0.29977652430534363 Test-Acc:0.8732 Train-Err:0.1565413773059845 Train-Acc:0.981
I:180 Test-Err:0.299514502286911 Test-Acc:0.8687 Train-Err:0.1744350641965866 Train-Acc:0.976
I:190 Test-Err:0.2970266342163086 Test-Acc:0.8672 Train-Err:0.15915338695049286 Train-Acc:0.987
I:200 Test-Err:0.30192652344703674 Test-Acc:0.8683 Train-Err:0.1647544801235199 Train-Acc:0.979
I:210 Test-Err:0.29446855187416077 Test-Acc:0.8689 Train-Err:0.1644645482301712 Train-Acc:0.981
I:220 Test-Err:0.2940334379673004 Test-Acc:0.8697 Train-Err:0.15401779115200043 Train-Acc:0.984
I:230 Test-Err:0.30237460136413574 Test-Acc:0.8718 Train-Err:0.16437652707099915 Train-Acc:0.987
I:240 Test-Err:0.2888593077659607 Test-Acc:0.8676 Train-Err:0.1494794636964798 Train-Acc:0.984
I:250 Test-Err:0.29642513394355774 Test-Acc:0.8645 Train-Err:0.1559772938489914 Train-Acc:0.984
I:260 Test-Err:0.29496461153030396 Test-Acc:0.8687 Train-Err:0.16007021069526672 Train-Acc:0.984
I:270 Test-Err:0.29296693205833435 Test-Acc:0.8679 Train-Err:0.14579840004444122 Train-Acc:0.988
I:280 Test-Err:0.3053768277168274 Test-Acc:0.8642 Train-Err:0.15551437437534332 Train-Acc:0.984
I:290 Test-Err:0.299437016248703 Test-Acc:0.8631 Train-Err:0.15237420797348022 Train-Acc:0.978
"""

