from mintorch import Tensor, nn, optim


import sys,random,math
from collections import Counter
import numpy as np
import re

#  1 Mary moved to the bathroom.
#  2 John went to the hallway.
#  3 Where is Mary? 	bathroom	1
#  4 Daniel went back to the hallway.
#  5 Sandra moved to the garden.
#  6 Where is Daniel? 	hallway	4
#  7 John moved to the office.
#  8 Sandra journeyed to the bathroom.
#  9 Where is Daniel? 	hallway	4
#  ...
#
#[['-', 'mary', 'moved', 'to', 'the', 'bathroom'], 
# ['-', 'john', 'went', 'to', 'the', 'hallway'], 
# ['-', 'where', 'is', 'mary?', 'bathroom', '1'],
#  ...]


f = open('tasksv11_en_qa1_single-supporting-fact_train.txt','r')
raw = f.readlines()
f.close()



tokens = list()
for line in raw[0:1000]:
    ##   todo - more clean-up? remove ? or ref numbers from targets or such?
    ## clean-up spaces (incl. tabs)
    line = line.lower()
    line = line.replace("\n","")    ##.replace(".","")
    line = re.sub(r'\s+', ' ', line)
    tokens.append(line.split( " " )[1:])

new_tokens = list()
for line in tokens:
    new_tokens.append(['-'] * (6 - len(line)) + line)  ## add padding if less than 6 tokens

tokens = new_tokens
print( "\ntokens:", tokens )

vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)
print( "\nvocab:", vocab )


word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i
    
def words2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx

indices = list()
for line in tokens:
    idx = list()
    for w in line:
        idx.append(word2index[w])
    indices.append(idx)

data = np.array(indices)
print( "\ndata:", data, data.shape )   ## 1000,6)



embed = nn.Embedding(vocab_size=len(vocab),dim=16)
model = nn.RNNCell(n_inputs=16, n_hidden=16, n_output=len(vocab))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(parameters=model.parameters() + embed.parameters(), lr=0.05)


for iter in range(1000):
    batch_size = 100
    total_loss = 0
    
    hidden = model.init_hidden(batch_size=batch_size)

    for t in range(5):   # 0,1,2,3,4
        input = Tensor(data[0:batch_size,t])
        rnn_input = embed(input=input)
        output, hidden = model(input=rnn_input, hidden=hidden)

    target = Tensor(data[0:batch_size,t+1])    
    loss = criterion(output, target)
  
    ## learn
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.data
    if(iter % 200 == 0):
        p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
        print("Loss:",total_loss / (len(data)/batch_size),"% Correct:",p_correct)



batch_size = 1
hidden = model.init_hidden(batch_size=batch_size)
for t in range(5):
    input = Tensor(data[0:batch_size,t])
    rnn_input = embed(input=input)
    output, hidden = model(input=rnn_input, hidden=hidden)

target = Tensor(data[0:batch_size,t+1])    
loss = criterion(output, target)

ctx = ""
for idx in data[0:batch_size][0][0:-1]:
    ctx += vocab[idx] + " "
print("Context:",ctx)
print("True:",vocab[target.data[0].astype(int)])
print("Pred:", vocab[output.data.argmax()])

# Context: - mary moved to the 
# True: bathroom.
# Pred: office.

print("bye")