from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.callbacks import History
from keras import regularizers
import os, sys
f = open(sys.argv[1],'r')
datas = f.readlines()[1:]
Y_train = [] 
X_train = []
n = 0
for data in datas:
    X_train.append([])
    label, pixels = data.split(',')
    pixels = list(map(float,pixels.split(' ')))
    #X_train[n] = pixels.reshape((48,48,1))
    X_train[n] = pixels
    Y_train.append(float(label))
    n += 1

#### end 
X_train = np.array(X_train) / 255.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_train = X_train.reshape(len(X_train), 48, 48, 1)

Y_train = np.array(Y_train)
Y_train = np.eye(7)[list(map(int,Y_train))]
### validation set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)


## Initialize
model =  Sequential()
## add CNN
model.add(Conv2D(32,(3, 3), input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(BatchNormalization())   # add 5

## add DNN
#model.add(Dense(106, kernel_initializer="uniform", use_bias=True,activation = 'tanh'))  # add 5
model.add(Dense(128, kernel_initializer="uniform", use_bias=True, activation = 'relu'))
model.add(Dropout(0.6))
model.add(BatchNormalization())
model.add(Dense(256, kernel_initializer="uniform", use_bias=True, activation = 'relu'))
model.add(Dropout(0.6))
model.add(BatchNormalization())
model.add(Dense(512, kernel_initializer="uniform", use_bias=True,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, kernel_initializer="uniform", use_bias=True,activation = 'softmax'))
## compile loss function
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
### checkpoint function
checkpoint = ModelCheckpoint("output/best.h5",monitor='val_acc',verbose=1,save_best_only=True,mode='max')
### call back function
call = EarlyStopping(monitor='val_loss', min_delta=0,patience=10)
### batch_print_callback
call_back_list = [checkpoint ,call]

### fit
history = model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=150, epochs=64, callbacks=call_back_list)

from keras.models import load_model
model.save('./CNN_model_hw.h5')

