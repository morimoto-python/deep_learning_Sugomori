# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 07:53:02 2018

@author: morim
"""

import numpy as np
import matplotlib.pyplot as plt


rng= np.random.RandomState(123)

d = 2 #データの次元
N = 10 #各パターンのデータ数
mean = -5 #ニューロンが発火するデータの平均値

x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])

x = np.concatenate((x1, x2), axis= 0)

#生成したデータをパーセプトロンで分類
w = np.zeros(d)
b = 0

def y(x): #出力 y=f(w*x + b)を表現
    return step(np.dot(w, x) + b)

def step(x): #ステップ関数
    return 1 * (x > 0)

def t(i): #パラメーターを更新するための出力値
    if i < N:
        return 0
    else:
        return 1

# 誤り訂正学習法
while True:
    classified = True
    for i in range(N*2):
        delta_w= (t(i) - y(x[i]))*x[i]
        delta_b= (t(i) - y(x[i]))
        w += delta_w
        b += delta_b
        classified *= all(delta_w == 0)*(delta_b == 0) #all(int):全ての要素がTrueの時にTrueを返す
    if classified:
        break
print("wの予測値：{}".format(w))
print("bの予測値：{}".format(b))

print(x)

X= np.linspace(-7,3,100)
Y= []
for i in X:
    Y.append(w[0]*i + w[1]*i + b)
 
plt.plot(X,Y)
plt.show()
        
        
        
        
    