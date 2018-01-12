from keras.models import load_model
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.cluster import KMeans
import sys
### main
encoder = load_model('./encoder.h5')

X = np.load(sys.argv[1])
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))

encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

### cluster and write data
import pandas as pd
# get test cases
f = pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

# predict
o = open(sys.argv[3],'w')
o.write("ID,ANS\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1 # two images are in the same cluster
    else:
        pred = 0 # two images are not in the same cluster
    o.write("{},{}\n".format(idx,pred))
o.close()
