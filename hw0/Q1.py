file = input()
f = open(file,'r').read()
strings = f.split(' ')
set_w = set(strings)
n = 0
data2=[]
data3=[]
data4=[]
for word in set_w:
    data2.append([word,strings.count(word)])
    
for i in strings:
    if i not in data3:
        data3.append(i)

for i in range(len(data3)):
    for j in range(len(data2)):
        if data3[i]==data2[j][0]:
            data4.append([data3[i],data2[j][1]])

data4[-1][0] = data4[-1][0].rstrip('\n')

with open('Q1.txt','w') as f2:
    for i in range(len(data4)):
        f2.write(str(data4[i][0])+' '+str(i)+' '+str(data4[i][1]))
        if i < len(data4)-1:
            f2.write('\n')
