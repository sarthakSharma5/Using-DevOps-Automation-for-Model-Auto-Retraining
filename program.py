#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.utils import np_utils


# In[3]:


import struct
import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
x_train = read_idx("./fashion/train-images-idx3-ubyte")
y_train = read_idx("./fashion/train-labels-idx1-ubyte")
x_test = read_idx("./fashion/t10k-images-idx3-ubyte")
y_test = read_idx("./fashion/t10k-labels-idx1-ubyte")

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Let's count the number columns in our hot encoded matrix 
print ("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


# In[4]:


l2_reg=0

model = Sequential()

# 1st Conv Layer : input layer
model.add(Conv2D(96, (5, 5), input_shape=input_shape, padding='same', kernel_regularizer=l2(l2_reg)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Conv Layer
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

print(model.summary())


# In[5]:


# func to add CRP layers
def addCRP(filters , fSize , pSize):
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters, (fSize, fSize), padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (pSize, pSize) ))
    
# func to add flattening layer
def addFlat() :
    model.add(Flatten())
    
# func to add FC layers
def addFC(n) :
    model.add(Dense(n))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


# In[18]:


import layer as ly
n_layer = ly.n_layers()

def tweak_crp() :
    for i in range(0, n_layer[0]) :
        crp = ly.crpdata()
        addCRP( crp[0], crp[1], crp[2] )
    
def tweak_fc() :
    for i in range(0, n_layer[1]) :
        fc = ly.fcdata()
        addFC( fc )
        
# control file

def controller() :
    control = open('control.txt','r')
    c = int(control.read())
    return c

# make code tweak-able = true
def control_o() :
    cc = open('control.txt','w')
    cc.write(str(1))
    cc.close()

# make code tweak-able = false
def control_z() :
    cc = open('control.txt','w')
    cc.write(str(0))
    cc.close()
    
control = controller()

if control == 1 :
    accStd = open('accuracy.txt','r')
    accuracy = float(accStd.read())
    accStd.close()
    if accuracy < 0.92 :
        tweak_crp()
        addFlat()
        tweak_fc()
        control_o()
    else :
        control_z()
        
n_layer


# In[7]:


# one extra FC Layer
addFC(4096)

# Output Layer
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))


# In[8]:


print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer=RMSprop(), metrics = ['accuracy'])


# In[9]:


# Training Parameters
batch_size = 128
epochs = 3

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
model.save("myCNN_model.h5")

# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

n_acc = scores[1]


# In[10]:


# oldaccuracy = open('C:/Users/KIIT/Desktop/MLOps/MLOps/Model to detect/accuracy.txt','r')
oldac = open('accuracy.txt','r')
o_acc = float(oldac.read())
oldac.close()

if n_acc > o_acc :
    # accuracyStored = open('C:/Users/KIIT/Desktop/MLOps/MLOps/Model to detect/accuracy.txt','w')
    acStd = open('accuracy.txt','w')
    acStd.write(str(n_acc))
    acStd.close()
    control_o()
    if n_acc > 0.80 :
        control_z()


# In[ ]:




