# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:11:28 2018

@author: morim
"""

"""
TensorFlowによる実装の流れ
1.モデルの定義
2.誤差関数の定義
3.最適化手法の定義
4.セッションの初期化
5.学習
"""

import numpy as np
import tensorflow as tf 


#重みとバイアスの初期化（要素0の多次元ベクトルの生成）
w = tf.Variable(tf.zeros([2 ,1]))
b = tf.Variable(tf.zeros([1]))

#シグモイド関数モデルの実装

"""
#　TensorFlowを使わない場合
def y(x):
    return sigmoid(np.dot(w, x) + b)

def sigmoid(p):
    return 1/(1 + np.exp(-p))

"""

#　TensorFlowを使う場合
x= tf.placeholder(tf.float32, shape=[None, 2])
t= tf.placeholder(tf.float32, shape=[None, 1])
y= tf.nn.sigmoid(tf.matmul(x, w) + b)

#交差エントロピー誤差関数とそれを最小化する勾配降下法
cross_entropy= - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy) #0.1は学習率

prediction= tf.equal(tf.to_float(tf.greater(y, 0.5)), t) #yが0.5以上で発火する

#学習用のデータの準備
X= np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y= np.array([[0], [1], [1], [1]])

# TFではセッションというデータのやり取りの中で計算が行われる
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#ORゲートの学習
for epoch in range(200):
    sess.run(train_step, feed_dict={
            x: X,
            t: Y
    })
    
#学習結果の確認
classified= prediction.eval(session= sess, feed_dict={
        x: X,
        t: Y
})

print(classified)

#各入力に対する出力確率
prob= y.eval(session= sess, feed_dict={
        x: X,
        t: Y
})
    
print(prob)
print("w:", sess.run(w))
print("b:", sess.run(b))
print()
print("classified:\n{}".format(classified))
print()
print("output probability:\n{}".format(prob))












  