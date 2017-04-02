import numpy as np
import sys
import csv
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')

def main():
    X = np.loadtxt(sys.argv[1], dtype = float, delimiter = ',',
                   skiprows = 1)
    Y = np.loadtxt(sys.argv[2], dtype = int)
    X_test = np.loadtxt(sys.argv[3], dtype = int, delimiter = ',',
                   skiprows = 1)
    X = np.append(X, np.ones((X.shape[0], 1)), axis = 1)
    X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis = 1)

    #X_tra = np.loadtxt('data/X_tra', dtype = float, delimiter = ',')
    #X_val = np.loadtxt('data/X_val', dtype = float, delimiter = ',')
    #X_tra = np.append(X_tra, np.ones((X_tra.shape[0], 1)), axis = 1)
    #X_val = np.append(X_val, np.ones((X_val.shape[0], 1)), axis = 1)

    normIndex = [0,1,3,4,5]
    X_mean = np.zeros(len(normIndex))
    X_std = np.zeros(len(normIndex))
    for i in range(len(normIndex)):
        X_mean[i] = X[:, normIndex[i]].mean()
        X_std[i] = X[:, normIndex[i]].std()

    for i in range(len(normIndex)):
        #X_tra[:, normIndex[i]] = (X_tra[:, normIndex[i]] - X_mean[i]) / X_std[i]
        #X_val[:, normIndex[i]] = (X_val[:, normIndex[i]] - X_mean[i]) / X_std[i]
        X_test[:, normIndex[i]] = (X_test[:, normIndex[i]] - X_mean[i]) / X_std[i]
        X[:, normIndex[i]] = (X[:, normIndex[i]] - X_mean[i]) / X_std[i]

    #Y_tra = np.loadtxt('data/Y_tra', dtype = int)
    #Y_val = np.loadtxt('data/Y_val', dtype = int)

    featureIndex = np.loadtxt('featureOrder62', dtype = int)
    #featureIndex = np.array(list(range(107)))
    #featureIndex = np.array(list(range(100)) + [101] + list(range(103,107)))
    
    #X_tra = X_tra[:, featureIndex]
    #X_val = X_val[:, featureIndex]
    X_test = X_test[:, featureIndex]
    X = X[:, featureIndex]

    w = gradientDescent(X, Y, 0.01, 100, 0)
    #np.savetxt('data/featureOrder', np.absolute(w).argsort(), '%s')
    #np.savetxt('data/weights1', w, '%s')

    #print('train: ', accuracy(X_tra, w, Y_tra))
    #print('val: ', accuracy(X_val, w, Y_val))
    #print('total: ', accuracy(X, w, Y))
    xList = []
    yList = []
    for i in range(X_test.shape[0]):
        z = sigmoid(X_test[i].dot(w))
        if z > 0.9:
            xList.append(X_test[i])
            yList.append(1)
        elif z < 0.1:
            xList.append(X_test[i])
            yList.append(0)

    newX = np.vstack((np.array(xList), X))
    newY = np.hstack((np.array(yList), Y))

    w = gradientDescent(newX, newY, 0.01, 100, 0)

    #print('train: ', accuracy(X_tra, w, Y_tra))
    #print('val: ', accuracy(X_val, w, Y_val))
    #print('total: ', accuracy(X, w, Y))
 
    



    ans = (sigmoid(X_test.dot(w)) > 0.5).astype(int)
    with open(sys.argv[4], 'w') as result:
        myWriter = csv.DictWriter(result, fieldnames = ['id', 'label'])
        myWriter.writeheader()
        for i in range(ans.shape[0]):
            myWriter.writerow({'id': (i + 1), 'label': ans[i]})


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(X, w, Y):
    Y_hat = (sigmoid(X.dot(w)) > 0.5).astype(int)
    return np.sum(Y_hat == Y) / Y.shape[0]

def gradientDescent(X, y, eta, iteration, lamda):
    w = np.zeros(X.shape[1])
    cost_history = [accuracy(X, w, y)]
    batch = 32
    batchNum = X.shape[0] // batch
    np.random.seed(0)
    index = np.arange(X.shape[0])
    for i in range(iteration):
        #if i % 10 == 0:
            #print(i)
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
        for j in range(batchNum):
           Xbatch = X[j*batch:(j+1)*batch]
           ybatch = y[j*batch:(j+1)*batch]
           k = np.zeros(X.shape[1])
           k[0:-2] = w[0:-2]
           grad = np.dot(Xbatch.T, (sigmoid(Xbatch.dot(w)) - ybatch)) + lamda * k  
           grad = grad / np.linalg.norm(grad)
           w = w - eta * grad 
        acc = accuracy(X, w, y)
        if acc < cost_history[-1]:
            eta = eta / 2

        cost_history.append(acc)

    #plt.plot(range(len(cost_history)), cost_history)
    #plt.show()
    return w

if __name__ == '__main__':
    main()
