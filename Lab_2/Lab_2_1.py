from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os



w = np.zeros((25))
D = None
Y0 = np.array([])

α = 0.2
β = -0.4
σ = lambda x: 1 if x > 0 else 0
 
def f(x):
    s = β + np.sum(x @ w)
    return σ(s)
 
def train():
    global w
    _w = w.copy()
    for x, y in zip(D, Y0):
        w += α * (y - f(x)) * x
    return (w != _w).any()


# Функция, открывающая все изображения в указанной папке,
# вычисляющая их ветор и ожидаемый ответ нейронной сети
def get_imgs(path):
  for name in os.listdir(path):
    img = plt.imread('{}{}'.format(path, name))
    # Преобразуем трехмерный массив в одномерный
    img = img.flatten()
    # Оставляем только каждый третий элемент (изображение ч/б)
    vec = img[np.mod(np.arange(img.size), 3) == 0]
    yield name, vec, int(name.split('_')[1].split('.')[0])

          
path = 'task_1/letters/'
for _, vec, result in get_imgs(path):
  Y0 = np.append(Y0, result)
  if D is None:
    D = vec
  else:
    D = np.vstack((D, vec))
 
print(D)
print(Y0)
      
while train():
    print(w)


path = 'task_1/test/'
for name, vec, result in get_imgs(path):
  print(name, f(vec), result)
