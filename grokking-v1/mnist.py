##
#  helper module for preparing mnist data
#    note - returns only first 1000 (o)

import numpy as np
from keras.datasets import mnist


def load_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, 
                    y_train[0:1000])

  one_hot_labels = np.zeros((len(labels),10))
  for i,l in enumerate(labels):
    one_hot_labels[i][l] = 1
  labels = one_hot_labels

  test_images = x_test.reshape(len(x_test),28*28) / 255
  test_labels = np.zeros((len(y_test),10))
  for i,l in enumerate(y_test):
    test_labels[i][l] = 1

  return (images, labels),(test_images,test_labels)


### todo/fix -  use one_hot or one_hot_encoded flag or such - why? why not?
##                  instead of _v2
def load_data_v2():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, 
                    y_train[0:1000])


  test_images = x_test.reshape(len(x_test),28*28) / 255

  return (images, labels),(test_images,y_test)
