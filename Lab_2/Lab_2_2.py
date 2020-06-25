from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os



w = np.zeros((5, 25))
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


def letter_to_bin(l):
  s = np.array(list(np.binary_repr(ord(l) - ord('a'))), dtype=float)
  if len(s) < 5:
    s = np.append(np.zeros(5 - len(s)), s)
  return s

def bin_to_letter(b):
  s = chr(int(b, 2) + ord('a'))
  return s


# Функция, открывающая все изображения в указанной папке,
# вычисляющая их ветор и ожидаемый ответ нейронной сети
def get_imgs(path):
  for name in os.listdir(path):
    img = plt.imread('{}{}'.format(path, name))
    # Преобразуем трехмерный массив в одномерный
    img = img.flatten()
    # Оставляем только каждый третий элемент (изображение ч/б)
    vec = img[np.mod(np.arange(img.size), 3) == 0]
    yield name, vec, letter_to_bin(name.split('.')[0])

       
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
    print('File name: {}  Net answer (adapted): {}  Net answer (original): {}'.format(name, bin_to_letter(answ), answ))



