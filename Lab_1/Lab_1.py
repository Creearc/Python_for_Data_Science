import numpy as np

# Читаем данные из файла созданного в Excel
x = np.loadtxt(delimiter=';', fname='Lab_1_info.csv') 
print(x)
print()

# Находим минимальное и максимальное значения
min_elem = np.min(x)
max_elem = np.max(x)

# Меняем местами максимальные и минимальные значения
# Чтобы значения не потерлись сначала записываем вместо минимальных элементов максимальное значение увеличенное на единицу
x[x == min_elem] = max_elem + 1
x[x == max_elem] = min_elem
x[x == max_elem + 1] = max_elem

print(x)

