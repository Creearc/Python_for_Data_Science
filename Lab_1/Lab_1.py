import numpy as np

x = np.loadtxt(delimiter=';', fname='Lab_1_info.csv')
print(x)
print()

min_elem = np.min(x)
max_elem = np.max(x)

x[x == min_elem] = max_elem + 1
x[x == max_elem] = min_elem
x[x == max_elem + 1] = max_elem

print(x)

