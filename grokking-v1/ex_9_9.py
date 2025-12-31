##########
#  mintorch version of ex_9_9.py
#     adds cross_entropy with integer labels (NOT one-hot encoded)

from mintorch import Tensor, nn, optim

import sys, numpy as np
import mnist

(images, labels), (test_images, test_labels) = mnist.load_data_v2()


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



criterion = nn.CrossEntropyLoss()
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
            correct_cnt += int(np.argmax(y_hat.data[k:k+1]) == labels[batch_start+k]) 
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
        test_error += loss.data
        test_correct_cnt += int(np.argmax(y_hat.data) == test_labels[i])
    if(j % 10 == 0):
        sys.stdout.write("\n"+ 
         f"I:{j}" + 
         f" Test-Err: {test_error/len(test_images)}" +
         f" Test-Acc:{test_correct_cnt/len(test_images)}" +
         f" Train-Err: {error/len(images)}" +
         f" Train-Acc:{correct_cnt/len(images)}")


print("\nbye")



"""
I:0 Test-Err: 1.9935849905014038 Test-Acc:0.619 Train-Err: 0.021619301289319992 Train-Acc:0.48
I:10 Test-Err: 0.6338382363319397 Test-Acc:0.8256 Train-Err: 0.005754734855145216 Train-Acc:0.863
I:20 Test-Err: 0.4936503469944 Test-Acc:0.854 Train-Err: 0.003771485760807991 Train-Acc:0.915
I:30 Test-Err: 0.4509413540363312 Test-Acc:0.8632 Train-Err: 0.0029342968482524157 Train-Acc:0.927
I:40 Test-Err: 0.43979814648628235 Test-Acc:0.865 Train-Err: 0.00246159709058702 Train-Acc:0.944
I:50 Test-Err: 0.4330494701862335 Test-Acc:0.8676 Train-Err: 0.0019566346891224384 Train-Acc:0.958
I:60 Test-Err: 0.43747562170028687 Test-Acc:0.8673 Train-Err: 0.00164832326117903 Train-Acc:0.964
I:70 Test-Err: 0.44234493374824524 Test-Acc:0.8667 Train-Err: 0.0014697157312184572 Train-Acc:0.967
I:80 Test-Err: 0.4508263170719147 Test-Acc:0.8664 Train-Err: 0.001101550180464983 Train-Acc:0.979
I:90 Test-Err: 0.45934808254241943 Test-Acc:0.8643 Train-Err: 0.0010338976280763745 Train-Acc:0.982
I:100 Test-Err: 0.4680449366569519 Test-Acc:0.8651 Train-Err: 0.0008650710806250572 Train-Acc:0.988
I:110 Test-Err: 0.47563427686691284 Test-Acc:0.865 Train-Err: 0.0008037285297177732 Train-Acc:0.987
I:120 Test-Err: 0.49006387591362 Test-Acc:0.8606 Train-Err: 0.0006887253257445991 Train-Acc:0.993
I:130 Test-Err: 0.48823970556259155 Test-Acc:0.8642 Train-Err: 0.0005751494318246841 Train-Acc:0.997
I:140 Test-Err: 0.5006069540977478 Test-Acc:0.8616 Train-Err: 0.0004976027412340045 Train-Acc:0.997
I:150 Test-Err: 0.5056841969490051 Test-Acc:0.8628 Train-Err: 0.0004906351678073406 Train-Acc:0.997
I:160 Test-Err: 0.5172780156135559 Test-Acc:0.8611 Train-Err: 0.0005001194076612592 Train-Acc:0.994
I:170 Test-Err: 0.523354709148407 Test-Acc:0.8614 Train-Err: 0.0004382041224744171 Train-Acc:0.998
I:180 Test-Err: 0.5289713740348816 Test-Acc:0.8617 Train-Err: 0.0004189533065073192 Train-Acc:0.995
I:190 Test-Err: 0.533804714679718 Test-Acc:0.8621 Train-Err: 0.00031440096790902317 Train-Acc:0.997
I:200 Test-Err: 0.5438050031661987 Test-Acc:0.8613 Train-Err: 0.00034222903195768595 Train-Acc:0.995
I:210 Test-Err: 0.5455694198608398 Test-Acc:0.8606 Train-Err: 0.00031357695115730166 Train-Acc:0.997
I:220 Test-Err: 0.5513232350349426 Test-Acc:0.86 Train-Err: 0.00031457163277082145 Train-Acc:0.999
I:230 Test-Err: 0.5552244782447815 Test-Acc:0.8612 Train-Err: 0.0002880828396882862 Train-Acc:0.999
I:240 Test-Err: 0.5640991926193237 Test-Acc:0.8606 Train-Err: 0.00030256278114393353 Train-Acc:0.997
I:250 Test-Err: 0.5686495304107666 Test-Acc:0.8596 Train-Err: 0.0002717635943554342 Train-Acc:1.0
I:260 Test-Err: 0.5732606649398804 Test-Acc:0.8602 Train-Err: 0.0002849621814675629 Train-Acc:0.994
I:270 Test-Err: 0.5746797919273376 Test-Acc:0.8608 Train-Err: 0.0002439698000671342 Train-Acc:0.998
I:280 Test-Err: 0.5808190703392029 Test-Acc:0.8586 Train-Err: 0.00019625779532361776 Train-Acc:0.999
I:290 Test-Err: 0.5871406197547913 Test-Acc:0.8591 Train-Err: 0.00022504478693008423 Train-Acc:0.998
"""