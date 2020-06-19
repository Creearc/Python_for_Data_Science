import numpy as np

x = np.zeros((10, 10))
x = np.ones((10, 10))
x = np.eye(10, 10)
x = np.random.randint(100, size=(10, 10))

np.savetxt(delimiter=';', fname='zeros', fmt='% 1.1i', X=x)

y = np.loadtxt(delimiter=';', fname='zeros')
