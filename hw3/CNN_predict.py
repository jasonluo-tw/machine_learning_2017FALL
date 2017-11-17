import csv
from keras.models import load_model
import numpy as np
import sys
model = load_model('./CNN_model_hw.h5')
f2 = open(sys.argv[1],'r')
datas = f2.readlines()[1:]
X_test = []
n = 0
for data in datas:
    X_test.append([])
    id, pixels = data.split(',')
    pixels = np.array(list(map(float, pixels.split(' '))))
    X_test[n] = pixels
    n += 1

X_test = np.array(X_test) / 255.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
X_test = X_test.reshape(len(X_test),48,48,1)

#print(X_test[:10])
## Predict
results = model.predict(X_test)
print(results.shape)
ans =[]
ans.append(['id','label'])
for i in range(results.shape[0]):
  ans.append([i,np.argmax(results[i,:])])

with open(sys.argv[2],'w+') as f:
  s = csv.writer(f,delimiter=',',lineterminator='\n')
  for i in range(len(ans)):
   s.writerow(ans[i])

