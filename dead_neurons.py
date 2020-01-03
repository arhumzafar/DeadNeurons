# Here we will build an inexperienced, random classifier

import numpy as np 

# Let's define some variables
# "x" will be a 4-dimensional, zero mean random value
# "y" will be the sign of x's mean

x = np.random.rand(320,4) - .5
y = np.zeros((320,1))

for i in range(x.shape[0]):
    y[i] = int(np.mean(x[i,:]) > 0)