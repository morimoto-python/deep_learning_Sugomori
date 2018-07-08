# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 15:15:38 2018

@author: morim
"""

import os
import numpy as np

os.chdir("C:\\Users\\morim\\OneDrive\\ドキュメント\Python\\git_hub\\deep_learning_Sugomori")


#ベクトルについて
a= np.array([0,2,3])
b= np.array([2,8,3])

print(a+b) #ベクトルの和
print(3*a) #ベクトルのスカラー倍
print(a*b) #ベクトルの要素積

print(np.dot(a,b)) #ベクトルの内積


#行列について
print("\n")
c= np.array([[1,3,1],[5,2,6]]) #2*3行列
d= np.array([[5,2,5],[1,2,0]]) #2*3行列

print(c+d)
print(5*c)
print(c*d) #行列の要素積

#　数学でいう行列の積はk*l行列とm*n行列においてl=mのときのみ可能なので、今回の例の場合はc行列を3*2転換する必要がある
C= c.T
print(C)
print(np.dot(C,d)) #行列の積

#配列・多次元配列の生成
print("\n")
print(np.zeros(5)) #全て0のベクトル
print(np.ones(7)) #全て1のベクトル

#　np.arange(start, end, step)
print(np.arange(5)) #範囲指定
print(np.arange(4,9))

#reshapeで行列（多次元配列）を作成
print(np.zeros(8).reshape(2,4))

#正方行列
print(np.identity(5))































