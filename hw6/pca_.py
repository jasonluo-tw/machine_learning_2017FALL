from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.zeros((30000,415))
img_data = []
for i in range(415):
    img_data.append(sys.argv[1]+'/%s.jpg' %i)
    img = io.imread(img_data[i])
    new_img = transform.resize(img, (100,100,3))
    new_img = new_img.flatten()
    X[:,i] = new_img

X_mean = np.mean(X, axis=1)

x = np.zeros(X.shape)
for i in range(X.shape[1]):
    x[:,i] = X[:,i] - X_mean
U, s, V = np.linalg.svd(x, full_matrices=False)

# 4 eigenface
y = X[:,int(sys.argv[2].split('.')[0])] - X_mean

uu = U[:,0:3]

y = np.dot(y,uu)

uuu = np.zeros(30000)
for i in range(3):
	#print(uuu)
	uuu += y[i] * uu[:,i]
# 轉換
new = uuu + X_mean
new -= np.min(new)
new /= np.max(new)
new = (new * 255).astype(np.uint8)
###
plt.figure(1, dpi=120)
plt.imshow(new.reshape(100,100,3))

plt.savefig("reconstruction.jpg")
