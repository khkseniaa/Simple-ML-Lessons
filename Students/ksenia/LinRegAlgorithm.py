
# coding: utf-8

# Author: K. Khmelevskaya

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Загрузка данных
dj = pd.read_csv('D&J-IND_101001_171001.txt')
gasp = pd.read_csv('GAZP_101001_171001.txt')
yndx = pd.read_csv('YNDX_101001_171001.txt')
print(dj.shape, gasp.shape, yndx.shape)


# Изначальная модель с 2мя параметрами
x1 = gasp['<CLOSE>']
x2 = gasp['<CLOSE>']
y = dj['<CLOSE>']

# Предварительная обработка данных
res1 = pd.merge(dj, gasp, on='<DATE>', suffixes=['_DJ', '_GAZP'])
res2 = pd.merge(res1, yndx, on='<DATE>', suffixes=['', '_YNDX'])
x1 = res2['<CLOSE>_DJ']
x2 = res2['<CLOSE>']
y = res2['<CLOSE>_GAZP']

# Нормализация данных
x1min, x1max, x2min, x2max, ymin, ymax = min(x1), max(x1), min(x2), max(x2), min(y), max(y)
x1 = (x1 - min(x1)) / (max(x1) - min(x1))
x2 = (x2 - min(x2)) / (max(x2) - min(x2))
y = (y - min(y)) / (max(y) - min(y))

datasets = [x1, x2]

# Класс, отвечающий за обучение парной линейной регрессии
class hypothesis(object):
    def __init__(self):
        self.theta = np.array([0, 0, 0])
    # Метод, возвращающий теоретический результат по переданным значениям факторов
    def apply(self, X1, X2):
        return self.theta[0] + self.theta[1] * X1 + self.theta[2] * X2
    # Функция ошибки
    def error(self, X1, X2, Y):
        return sum((self.apply(X1, X2) - Y)**2) / (2 * len(Y))
    # Метод, реализующий градиентный спуск
    def gradient_descent(self, x1, x2, y):
        i = 0
        m = len(x1)
        steps = []
        errors = []
        while(i < 100):
            y_ = hyp.apply(x1, x2)
            dJ0 = sum(y_ - y) / m
            dJ1 = sum((y_ - y)*x1) / m
            dJ2 = sum((y_ - y)*x2)/ m
            
            alpha = 0.9
            theta0 = self.theta[0] - alpha * dJ0
            theta1 = self.theta[1] - alpha * dJ1
            theta2 = self.theta[2] - alpha * dJ2
            self.theta = np.array([theta0, theta1, theta2])
            
            J = hyp.error(x1, x2, y)
            
            steps.append(i)
            errors.append(J)
            
            i += 1
        
        return (steps, errors)

# Заводим и обучаем нашу регрессию
hyp = hypothesis()

y_ = hyp.apply(x1, x2)

J = hyp.error(x1, x2, y)
print(J)

(steps, errors) = hyp.gradient_descent(x1, x2, y)

y_ = hyp.apply(x1, x2)

plt.plot(steps, errors)
plt.show()

# Денормализуем данные и параметры регрессии
x1 = x1 * (x1max - x1min) + x1min
x2 = x2 * (x2max - x2min) + x2min
y = y * (ymax - ymin) + ymin

theta1 = hyp.theta[1] * (ymax - ymin) / (x1max - x1min)
theta2 = hyp.theta[2] * (ymax - ymin) / (x2max - x2min)
theta0 = hyp.theta[0] * (ymax - ymin) + ymin - theta1*x1min - theta2*x1min
hyp.theta = np.array([theta0, theta1, theta2])
J = hyp.error(x1, x2, y)
print(J)


# Универсальная модель для любого количествтва параметров
# Для сравнения работы с предыдущим вариантом используются те же массивы данных
x1 = gasp['<CLOSE>']
x2 = gasp['<CLOSE>']
y = dj['<CLOSE>']

# Предварительная обработка данных
res1 = pd.merge(dj, gasp, on='<DATE>', suffixes=['_DJ', '_GAZP'])
res2 = pd.merge(res1, yndx, on='<DATE>', suffixes=['', '_YNDX'])
x1 = res2['<CLOSE>_DJ']
x2 = res2['<CLOSE>']
y = res2['<CLOSE>_GAZP']

# Нормализация данных
datasets = [x1, x2]
datasets = map(lambda x: (x - min(x)) / (max(x) - min(x)), datasets)
y = (y - min(y)) / (max(y) - min(y))

class hypothesis2(object):
    def __init__(self):
        self.theta = np.array([0]*(len(datasets) + 1))
    # Метод, возвращающий теоретический результат по переданным значениям факторов
    def apply(self, dataarr):
    # dataarr - массив из наборов данных
        func = self.theta[0]
        i = 1
        for el in dataarr:
            if i <= len(dataarr):
                func += self.theta[i] * el
            i += 1
        return func
        
    # Функция ошибки
    def error(self, dataarr, Y):
        return sum((self.apply(dataarr) - Y)**2) / (2 * len(Y))
    # Метод, реализующий градиентный спуск
    def gradient_descent(self, dataarr, y):
        i = 0
        m = len(dataarr[0])
        steps = []
        errors = []
        while(i < 100):
            y_ = hyp.apply(dataarr)
            dJ0 = sum(y_ - y) / m
            dJn = [dJ0]
            for el in dataarr:
                dJn.append(sum((y_ - y) * el)/m)
            alpha = 0.9
            pairs = zip(self.theta, dJn)
            pairs = map(lambda x: list(x), pairs)
            for pair in pairs:
                pair[0] -= alpha * pair[1]
            self.theta = np.array(map(lambda x: x[0], pairs))
            
            J = hyp.error(dataarr, y)
            steps.append(i)
            errors.append(J)
            
            i += 1
        
        return (steps, errors)

# Заводим и обучаем нашу регрессию
hyp = hypothesis2()

y_ = hyp.apply(datasets)

J = hyp.error(datasets, y)
print(J)

(steps, errors) = hyp.gradient_descent(datasets, y)

y_ = hyp.apply(datasets)

plt.plot(steps, errors)
plt.show()

# Денормализуем данные и параметры регрессии
for dataset in datasets:
    dataset = dataset * (max(dataset) - min(dataset)) + min(dataset)

y = y * (ymax - ymin) + ymin

pairs = zip(hyp.theta[1:], datasets)
pairs = map(lambda x: list(x), pairs)
coefs = 0
for el in pairs:
    el[0] = el[0] * (ymax - ymin) / (max(el[1]) - min(el[1]))
    coefs -= el[0]*min(el[1])
hyp.theta[1:] = np.array(map(lambda x: x[0], pairs))
hyp.theta[0] = hyp.theta[0] * (ymax - ymin) + coefs

J = hyp.error(datasets, y)
print(J)
