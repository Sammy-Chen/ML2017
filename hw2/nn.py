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
     
    normIndex = [0,1,3,4,5]
    X_mean = np.zeros(len(normIndex))
    X_std = np.zeros(len(normIndex))
    for i in range(len(normIndex)):
        X_mean[i] = X[:, normIndex[i]].mean()
        X_std[i] = X[:, normIndex[i]].std()
    for i in range(len(normIndex)):
        X_test[:, normIndex[i]] = (X_test[:, normIndex[i]] - X_mean[i]) / X_std[i]
        X[:, normIndex[i]] = (X[:, normIndex[i]] - X_mean[i]) / X_std[i]

    w1, w2 = gradientDescent(X, Y, 32, 0.01, 50, 0)
    a1, a2 = forward(X_test, w1, w2)
    ans = (a2 > 0.5).astype(int)
    with open(sys.argv[4], 'w') as result:
        myWriter = csv.DictWriter(result, fieldnames = ['id', 'label'])
        myWriter.writeheader()
        for i in range(ans.shape[0]):
            myWriter.writerow({'id': (i + 1), 'label': ans[i]})


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, w1, w2):
    a1 = sigmoid(X.dot(w1))
    a1[-1, :] = np.ones(X.shape[1])
    a2 = sigmoid(a1.dot(w2))
    return (a1, a2)

def accuracy(X, w1, w2, Y):
    a1, a2 = forward(X, w1, w2)
    return (np.sum((a2 > 0.5) == Y)) / Y.shape[0]

def backward(a1, a2, w2, X, Y):
    dw2 = a1.T.dot(a2 - Y)
    dw1 = X.T.dot(a1 * (1 - a1) * (a2 - Y).reshape((Y.shape[0], 1)) * w2) 
    return (dw1, dw2)

def gradientDescent(X, Y, batchSize, eta, epoach, lamda):
    featNum = X.shape[1]
    np.random.seed(0)
    w1 = 0.1 * np.random.randn(featNum, featNum) 
    w2 = 0.1 * np.random.randn(featNum) 
    g1 = w1 ** 2 
    g2 = w2 ** 2
    batchNum = X.shape[0] // batchSize 
    index = np.arange(X.shape[0])
    cost_history = [accuracy(X, w1, w2, Y)]
    for i in range(epoach):
        np.random.shuffle(index)
        X = X[index]
        Y = Y[index]
        for j in range(batchNum):
            Xbatch = X[j * batchSize: (j + 1) * batchSize]
            Ybatch = Y[j * batchSize: (j + 1) * batchSize]
            a1, a2 = forward(Xbatch, w1, w2)
            dw1, dw2 = backward(a1, a2, w2, Xbatch, Ybatch)
            w1 = w1 - eta / np.sqrt(g1 + np.exp(-8)) * dw1
            w2 = w2 - eta / np.sqrt(g2 + np.exp(-8)) * dw2
            g1 = g1 + w1 ** 2
            g2 = g2 + w2 ** 2

        if i % 5 == 0:
            acc = accuracy(X, w1, w2, Y)
            cost_history.append(acc)

    #plt.plot(range(len(cost_history)), cost_history)
    #plt.show()
    return (w1, w2)

if __name__ == '__main__':
    main()
