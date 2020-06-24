from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os



w = np.zeros((7, 25))
D = None
Y = None

α = 0.2
β = -0.4
σ = lambda x: 1 if x > 0 else 0
 
def f(x, i):
    s = β + np.sum(x @ w[i])
    return σ(s)
 
def train(i):
    global w
    _w = w[i].copy()
    for x, y in zip(D, Y[i]):
        w[i] += α * (y - f(x, i)) * x
    return (w[i] != _w).any()


def net(vec):
    s = ''
    for i in range(np.shape(w)[0]):
        s = '{}{}'.format(s, f(vec, i))
    return(s)


# Функция, открывающая все изображения в указанной папке,
# вычисляющая их ветор и ожидаемый ответ нейронной сети
def get_imgs(path):
  for name in os.listdir(path):
    img = plt.imread('{}{}'.format(path, name))
    # Преобразуем трехмерный массив в одномерный
    img = img.flatten()
    # Оставляем только каждый третий элемент (изображение ч/б)
    vec = img[np.mod(np.arange(img.size), 3) == 0]
    yield name, vec, np.array(list(np.binary_repr(ord(name.split('.')[0]))), dtype=float)

       
path = 'task_2/letters/'
for _, vec, result in get_imgs(path):
  if D is None:
    D = vec
    Y = result
  else:
    D = np.vstack((D, vec))
    Y = np.vstack((Y, result))

Y = np.swapaxes(Y, 0, 1)

print(D)
print(Y)


for i in range(np.shape(w)[0]):      
    while train(i):
        print(w[i])



path = 'task_2/letters/'
for name, vec, result in get_imgs(path):
    answ = net(vec)
    print('File name: {}  Net answer (adapted): {}  Net answer (original): {}'.format(name, chr(int(answ, 2)), answ))



