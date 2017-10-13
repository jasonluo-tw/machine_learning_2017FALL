import csv
import numpy as np
import math
filein = input()
fileout = input()
# read model
w = np.load('hw1_best.npy')

# read data

test_x = []
n_row = 0
text = open(filein ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x[:])
x_pm = test_x[:,9*9:9*10]
x_pm = np.concatenate((x_pm,x_pm**2), axis=1)
# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
#test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
x_pm = np.concatenate((np.ones((x_pm.shape[0],1)),x_pm), axis=1)



## answer
ans = []
for i in range(len(x_pm)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,x_pm[i])
    ans[i].append(a)

text = open(fileout, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

