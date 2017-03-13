import numpy as np
import csv
import sys

data = np.zeros((4320,24))
interval = 10 #hours

with open(sys.argv[1], 'r', encoding = 'big5') as trainData:
  i = 0
  for row in csv.reader(trainData):
    if row[2] == 'RAINFALL':
        data[i - 1] = np.array([(0 if x == 'NR' else x) 
                    for x in row[3:]], dtype = float)
    elif i is not 0:
        data[i - 1] = np.array(row[3:], dtype = float)
    i = i + 1
  
#ndata is the basic 18 features taken every 10 hours
ndata = np.zeros((18, 240 * 24))

for i in range(18):
    ndata[i] = data[i::18].flat

mdataNum = 480 - interval + 1
dataNum = 12 * mdataNum 
#conData is ndata taken from consecutive 10 hours 
conData = np.zeros((18+18+9*17, dataNum * 10))


for i in range(12):
    for j in range(mdataNum):
        conData[:18,(i*mdataNum+j)*interval:(i*mdataNum+j+1)*interval] = ndata[:,480*i+j:480*i+j+interval]

#conData = np.vstack((conData, conData ** 2))
conData[18:36] = conData[:18]**2
index = 0
for i in range(17):
    for j in range(i+1,18):
        conData[36+index] = conData[i] * conData[j]
        index = index + 1


"""#normalize feature
mu = conData.mean(1)
sigma = conData.std(1)
for i in range(36+9*17):
    for j in range(dataNum):
        if j % 10 is not 9:
            conData[i][j] = (conData[i][j] - mu[i]) / sigma[i]
"""#-----------------

featureIndex = np.array(list(range(18+18+9*17)))
selectIndex = (np.loadtxt('data/myselect', dtype = int))
featureNum = len(featureIndex)
pmIndex = 9

X = np.zeros((dataNum, featureNum * (interval - 1) + 1 + 1)) 
y = np.zeros(dataNum)

for i in range(dataNum):
    X[i][0] = conData[pmIndex, (i + 1) * interval - 1]
    X[i][1] = 1
    X[i][2:] = conData[featureIndex, i*interval:(i+1)*interval-1].flat

Xmean = X.mean(0)
Xstd = X.std(0)
X[:,2:] = (X[:,2:] - Xmean[2:]) / Xstd[2:]

np.random.seed(0)
np.random.shuffle(X)
"""
np.savetxt('data/total.X', X[:,1:], '%s', ',')
np.savetxt('data/total.y', X[:,0], '%s')

np.savetxt('data/val.X', X[4652:,1:], '%s', ',')
np.savetxt('data/val.y', X[4652:,0], '%s')
np.savetxt('data/train.X', X[:4652,1:], '%s', ',')
np.savetxt('data/train.y', X[:4652,0], '%s')
"""
np.savetxt('data/total.X', X[:,selectIndex+1], '%s', ',')
np.savetxt('data/total.y', X[:,0], '%s')

np.savetxt('data/val.X', X[4652:,selectIndex+1], '%s', ',')
np.savetxt('data/val.y', X[4652:,0], '%s')
np.savetxt('data/train.X', X[:4652,selectIndex+1], '%s', ',')
np.savetxt('data/train.y', X[:4652,0], '%s')



#--------------------------------------------------------------

tdata = np.zeros((4320, 9))

with open(sys.argv[2], 'r') as testData:
    i = 0
    for row in csv.reader(testData):
        if row[1] == 'RAINFALL':
            tdata[i,:9] = np.array([(0 if x == 'NR' else x) 
                        for x in row[2:]], dtype = float)
        else:
            tdata[i,:9] = np.array(row[2:], dtype = float)
        i = i + 1

tX = np.zeros((240, featureNum * 9 + 1)) 
for i in range(240):
    tX[i][0] = 1
    tX[i][1:18*9+1] = tdata[i*18:(i+1)*18,:].flat
    tX[i][18*9+1:18*18+1] = tX[i][1:18*9+1] ** 2
    index = 0
    for j in range(17):
        for k in range(j+1,18):
            tX[i][18*18+1+index*9:18*18+1+index*9 + 9] = tdata[i*18+j]*tdata[i*18+k]
            index = index + 1
tX[:,1:] = (tX[:,1:] - Xmean[2:]) / Xstd[2:]
np.savetxt('data/test.X', tX[:,selectIndex], '%s', ',')