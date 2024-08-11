import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

print("123")

data = loadmat('ex4data1.mat')

print(data)

X = data['X']
y = data['y']

print(X.shape)
print(y.shape)
# 400个纬度，1个结果
# 5000条case


# OneHot
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

print(y_onehot)
print(y_onehot.shape)

print(y[0])
print(y_onehot[0, :])
print(y[4999])
print(y_onehot[4999, :])


# sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 前向传播函数
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    # ？只有两层吗，看不懂
    return a1, z2, a2, z3, h


def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 没看懂
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)]
                                  , (hidden_size, (input_size + 1))))

    # 这个是什么，没看懂，为什么从 params 中取呢。
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):]
                                  , (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 如何逼近呢，获取值呢，没看懂。

    # compute the cost
    J = 0

    for i in range(m):
        first_tem = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_tem - second_term)

    J = J / m

    # K 直接被向量替换了
    return J


# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# 随机初始化完整网络参数大小的参数数组

params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

print(theta1.shape, theta2.shape)

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

print(a1.shape, z2.shape, z2.shape, z3.shape, h.shape)

costValue = cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

print(costValue)


# 增加了代价函数
def costWithPrice(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 没看懂
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)]
                                  , (hidden_size, (input_size + 1))))

    # 这个是什么，没看懂，为什么从 params 中取呢。
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):]
                                  , (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 如何逼近呢，获取值呢，没看懂。

    # compute the cost
    J = 0
    for i in range(m):
        first_tem = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_tem - second_term)

    J = J / m

    # add cost regularization term
    J += (float(learning_rate) / (2 * m)) \
         * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # K 直接被向量替换了
    return J


# 代价函数 with 正则化
costValue2 = costWithPrice(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

print(costValue2)


# sigmoid 梯度函数，怎么算出来的，吐血
# 看不懂
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# 反向传播，还是这样写，学得多点
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # 这是用来干嘛的
    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        # 完全看不懂，看了想睡
        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

print(J, grad.shape)


class TookTooLong(Warning):
    pass


import warnings
import time


class MinimizeStopper(object):
    def __init__(self, max_sec=10):
        self.max_sec = max_sec
        self.start = time.time()

    def __call__(self, xk):
        # callback to terminate if max_sec exceeded
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            # you might want to report other stuff here
            print("Elapsed: %.3f sec" % elapsed)


# 预测
from scipy.optimize import minimize

# minimize the objective function

# fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
#                 method='TNC', jac=True, options={'maxiter': 250})

# init stopper
minimize_stopper = MinimizeStopper()

fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={"disp": True, "maxfun": 250}
                )

print(fmin)

# 我们的总代价已经下降到0.5以下，这是算法正常工作的一个很好的指标。
# 为什么0.5是好的，没看懂。


## 谁TM 发现这种解析方式的，脑子有病吧
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

print(y_pred)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]

accuracy = (sum(map(int, correct))) / float(len(correct))

print(accuracy)
