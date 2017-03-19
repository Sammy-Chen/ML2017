import numpy as np
import sys
import csv

def main():
    X = np.loadtxt('total.X', dtype = float, delimiter = ',')
    y = np.loadtxt('total.y', dtype = float)
    X_test = np.loadtxt('test.X', dtype = float, delimiter = ',')

    lamda = 0
    w = gradientDescent(X, y, 0.1, 2000, lamda)

    ans = X_test.dot(w)
    with open(sys.argv[1], 'w') as result:
        myWriter = csv.DictWriter(result, fieldnames = ['id', 'value'])
        myWriter.writeheader()
        for i in range(240):
            myWriter.writerow({'id': 'id_' + str(i), 'value': ans[i]})

def gradientDescent(X, y, eta, iteration, lamda):
    w = np.zeros(X.shape[1])
    cost_history = [np.sqrt(((X.dot(w)-y)**2).mean())]
    batch = 32
    batchNum = X.shape[0] // batch
    for i in range(iteration):
        for j in range(batchNum):
           Xbatch = X[j*batch:(j+1)*batch]
           ybatch = y[j*batch:(j+1)*batch]
           grad = np.dot(Xbatch.T, (Xbatch.dot(w) - ybatch)) + lamda * w 
           grad = grad / np.linalg.norm(grad)
           w = w - eta * grad 

        rms = np.sqrt(((X.dot(w)-y)**2).mean())
        if rms > cost_history[-1]:
            eta = eta / 1.2
        cost_history.append(rms)

    return w



if __name__ == '__main__':
    main()
