import numpy as np
import csv
import sys
X = []
with open(sys.argv[3]) as f:
    row = csv.reader(f, delimiter =",")
    next(row,None)
    for r in row:
        X.append(list(map(float,r)))
        
X = np.array(X)

y = []
with open(sys.argv[4]) as f:
    row = csv.reader(f, delimiter =",")
    next(row,None)
    for r in row:
        y.append(list(map(float,r)))
        
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.fit_transform(X_val)
X = sc.fit_transform(X)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping

classfier = Sequential()

#hidden1
classfier.add(Dense(106, kernel_initializer="uniform", use_bias=True, input_shape=(106,)))
classfier.add(BatchNormalization())
classfier.add(Activation('relu'))
classfier.add(Dropout(0.7))


classfier.add(Dense(1, kernel_initializer="uniform", use_bias=True, activation="sigmoid"))

classfier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classfier.fit(X, y, batch_size=100, epochs=100)
print(classfier.evaluate(X,y))

x_test = []
with open(sys.argv[5]) as f:
    row = csv.reader(f, delimiter =",")
    next(row,None)
    for r in row:
        x_test.append(list(map(float,r)))
        
x_test = np.array(x_test)
x_test = sc.fit_transform(x_test)

pre = classfier.predict(x_test)
pre = pre > 0.5

ans = []
ans.append(["id", "label"])
for i in range(pre.shape[0]):
    ans.append([i+1,int(pre[i])])

with open(sys.argv[6],'w+') as f:
    s = csv.writer(f,delimiter=',',lineterminator='\n')
    for i in range(len(ans)):
        s.writerow(ans[i])
        

