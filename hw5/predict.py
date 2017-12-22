from keras.models import load_model
import pandas as pd
import numpy as np
import sys
def load_data(data_path, idendity):
    data = pd.read_csv(data_path, index_col=None)
    if idendity == 'train':
        userID = data['UserID'].values
        movieID = data['MovieID'].values
        label = data['Rating'].values
        return userID, movieID, label
    elif idendity == 'test':
        userID = data['UserID'].values
        movieID = data['MovieID'].values
        return userID, movieID

### main program
test_userID, test_movieID = load_data(sys.argv[1],'test')
model = load_model('./Model_MF500.h5')
result = model.predict([test_userID, test_movieID], verbose=1)

f = open(sys.argv[2], 'w')
f.write("TestDataID,Rating\n")
for i in range (result.shape[0]):
    if (result[i][0] > 5):
        f.write(str(i+1) + ",5\n")
    elif (result[i][0] < 1):
        f.write(str(i+1) + ",1\n")
    else:
        f.write(str(i+1) + "," + str(result[i][0]) + "\n")
f.close()
###
print('done predict')
