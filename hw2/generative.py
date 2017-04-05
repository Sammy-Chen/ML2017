import numpy as np
import sys
import csv

def main():
    X = np.loadtxt(sys.argv[1], dtype = float, delimiter = ',',
                   skiprows = 1)
    Y = np.loadtxt(sys.argv[2], dtype = int)
    X_test = np.loadtxt(sys.argv[3], dtype = int, delimiter = ',',
                   skiprows = 1)
    #X_tra = np.loadtxt('data/X_tra', dtype = float, delimiter = ',')
    #X_val = np.loadtxt('data/X_val', dtype = float, delimiter = ',')
    #Y_tra = np.loadtxt('data/Y_tra', dtype = int)
    #Y_val = np.loadtxt('data/Y_val', dtype = int)

    #normIndex = [0,1,3,4,5]
    normIndex = [0,3,4,5]
    X_mean = np.zeros(len(normIndex))
    X_std = np.zeros(len(normIndex))
    # change to X
    for i in range(len(normIndex)):
        X_mean[i] = X[:, normIndex[i]].mean()
        X_std[i] = X[:, normIndex[i]].std()

    for i in range(len(normIndex)):
        #X_tra[:, normIndex[i]] = (X_tra[:, normIndex[i]] - X_mean[i]) / X_std[i]
        #X_val[:, normIndex[i]] = (X_val[:, normIndex[i]] - X_mean[i]) / X_std[i]
        X_test[:, normIndex[i]] = (X_test[:, normIndex[i]] - X_mean[i]) / X_std[i]
        X[:, normIndex[i]] = (X[:, normIndex[i]] - X_mean[i]) / X_std[i]

    p_c1 = np.sum(Y) / Y.shape[0]
    px_c1 = np.zeros(X.shape[1])
    px_c2 = np.zeros(X.shape[1])
    mean_c = np.zeros((len(normIndex), 2))
    std_c = np.zeros((len(normIndex), 2))

    #bernoulliIndex = [2] + list(range(6, X_tra.shape[1]))
    bernoulliIndex = np.loadtxt('featureForGen', dtype = int)

    for i in bernoulliIndex:
        px_c1[i], px_c2[i] = pXgivenY(X[:, i], Y)

    for i in range(len(normIndex)):
        one_x = np.array([x for y, x in zip(Y, X[:, normIndex[i]]) if y == 1])
        zero_x = np.array([x for y, x in zip(Y, X[:, normIndex[i]]) if y == 0])
        mean_c[i][0] = np.mean(zero_x) 
        mean_c[i][1] = np.mean(one_x) 
        std_c[i][0] = np.std(zero_x)
        std_c[i][1] = np.std(one_x)

    likeli_c1, likeli_c2 = likelihood(X_test, normIndex, bernoulliIndex, mean_c, std_c, px_c1, px_c2) 
    likeli_c1 += np.log(p_c1) 
    likeli_c2 += np.log(1 - p_c1) 
    #print(np.sum((likeli_c1 > likeli_c2) == Y_tra) / Y_tra.shape[0])
    ans = (likeli_c1 > likeli_c2).astype(int)


    with open(sys.argv[4], 'w') as result:
        myWriter = csv.DictWriter(result, fieldnames = ['id', 'label'])
        myWriter.writeheader()
        for i in range(ans.shape[0]):
            myWriter.writerow({'id': (i + 1), 'label': ans[i]})

def gaussian(x, mu, std):
    return np.exp(- (x - mu) ** 2 / (2 * std**2 )) / np.sqrt(2 * np.pi * std ** 2)

def likelihood(X, index1, index2, mean, std, p_c1, p_c2):
    likeli_c1 = np.zeros(X.shape[0])
    likeli_c2 = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(len(index1)):
            likeli_c1[i] += np.log(gaussian(X[i, index1[j]], mean[j][1], std[j][1]) + np.exp(-8))
            likeli_c2[i] += np.log(gaussian(X[i, index1[j]], mean[j][0], std[j][0]) + np.exp(-8))
        for j in range(len(index2)):
            likeli_c1[i] += np.log(((1 - X[i,index2[j]]) * (1 - p_c1[j])) + X[i,index2[j]] * p_c1[index2[j]] + np.exp(-8) )
            likeli_c2[i] += np.log(((1 - X[i,index2[j]]) * (1 - p_c2[j])) + X[i,index2[j]] * p_c2[index2[j]] + np.exp(-8) )

    return (likeli_c1, likeli_c2)

def pXgivenY(x, y):
    n1_1 = np.sum((x + y) == 2)
    n0_0 = np.sum((x + y) == 0)
    n1 = np.sum(y == 1)
    n0 = np.sum(y == 0)
    return (n1_1 / n1, 1 - n0_0 / n0)


if __name__ == '__main__':
    main()
