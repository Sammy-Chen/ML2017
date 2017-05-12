import numpy as np
import sys

data = np.load(sys.argv[1])
test = np.loadtxt('my_model')
ans = np.zeros(200)
z = np.zeros(5)
for i in range(200):
    for j in range(5):
       x = (data[str(i)])[500*j:500*(j+1),:]
       z[j] = x.std(axis=0).mean()
    ans[i] = z[j].mean()
    mins = 50
    index = 0
    for k in range(60):
        if np.abs(ans[i] - test[k]) < mins:
            mins = np.abs(ans[i] - test[k])
            index = k
    ans[i] = np.log(index + 1)
    
with open(sys.argv[2], 'w') as f:
    f.write('SetId,LogDim\n')
    for i in range(200):
        f.write('{},{}\n'.format(i, ans[i]))
