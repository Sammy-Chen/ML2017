import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')
import numpy as np
import csv

def main():
    #rI = 0 * I 
    #rz = (np.linalg.inv(rX.T.dot(rX) + rI ).dot(rX.T).dot(ry))
    #print(np.sqrt(((rX.dot(rz)-ry)**2).mean()))
    #print(np.sqrt(((vX.dot(rz)-vy)**2).mean()))
    #w = (np.linalg.inv(X.T.dot(X) + rI ).dot(X.T).dot(y))
    #print(np.sqrt(((X.dot(w)-y)**2).mean()))

    X = np.loadtxt('data/total.X', dtype = float, delimiter = ',')
    #X_val = np.loadtxt('data/val.X', dtype = float, delimiter = ',')
    #X_train = np.loadtxt('data/train.X', dtype = float, delimiter = ',')
    y = np.loadtxt('data/total.y', dtype = float)
    #y_val = np.loadtxt('data/val.y', dtype = float)
    #y_train = np.loadtxt('data/train.y', dtype = float)
    X_test = np.loadtxt('data/test.X', dtype = float, delimiter = ',')

    I = np.eye(X.shape[1], dtype=int)
    I[0][0] = 0

    lamda = 0
    #w_lin = (np.linalg.inv(X_train.T.dot(X_train) + I * lamda).dot(X_train.T).dot(y_train))
    #w = gradientDescent(X_train, y_train, 0.1, 2000, lamda)
    #np.savetxt('data/wvalue', w, '%s')
    w_lin = (np.linalg.inv(X.T.dot(X) + I * lamda).dot(X.T).dot(y))
    w = gradientDescent(X, y, 0.1, 2000, lamda)
    #print('normal,train: ', np.sqrt(((X_train.dot(w_lin)-y_train)**2).mean()))
    #print('grad, train: ', np.sqrt(((X_train.dot(w)-y_train)**2).mean()))
    #print('normal,val: ', np.sqrt(((X_val.dot(w_lin)-y_val)**2).mean()))
    #print('grad, val: ', np.sqrt(((X_val.dot(w)-y_val)**2).mean()))
    print('grad, total: ', np.sqrt(((X.dot(w)-y)**2).mean()))
    print('normal, total: ', np.sqrt(((X.dot(w_lin)-y)**2).mean()))

    ans = X_test.dot(w)
    with open('res1.csv', 'w') as result:
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
        if i % 10 == 0:
            print(i)
        for j in range(batchNum):
           Xbatch = X[j*batch:(j+1)*batch]
           ybatch = y[j*batch:(j+1)*batch]
           grad = np.dot(Xbatch.T, (Xbatch.dot(w) - ybatch)) + lamda * w 
           #grad = np.dot(Xbatch.T, (Xbatch.dot(w) - ybatch)) + lamda * np.sign(w) 
           grad = grad / np.linalg.norm(grad)
           w = w - eta * grad 
           #rms = np.sqrt(((X.dot(w)-y)**2).mean())
           #if rms > cost_history[-1]:
               #eta = eta / 1.1

        rms = np.sqrt(((X.dot(w)-y)**2).mean())
        if rms > cost_history[-1]:
            eta = eta / 1.2
        cost_history.append(rms)

    #plt.plot(range(len(cost_history)), cost_history)
    #plt.show()
    #print(cost_history[-1])
    return w



if __name__ == '__main__':
    main()
