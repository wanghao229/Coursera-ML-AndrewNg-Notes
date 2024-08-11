import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "ex2data1.txt"

data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admiited'])

# print(data)
data.head()

positive = data[data['Admiited'].isin([1])]
negative = data[data['Admiited'].isin([0])]

print(positive)
print(negative)

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')

ax.legend()

ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


# plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


nums = np.arange(-10, 10, step=1)

print(nums)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(nums, sigmoid(nums), 'r')


# plt.show()

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

print(theta)

print(X.shape, theta.shape, y.shape)

print(cost(theta, X, y))


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    return error.T * X / len(X)


print(gradient(theta, X, y))

import scipy.optimize as opt

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)

print(cost(result[0], X, y))


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_min = np.matrix(result[0])

predictions = predict(theta_min, X)

print(predictions)

correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]

print(correct)

ratio = (sum(map(int, correct)) / len(correct))

print(ratio)

# 正则化逻辑回归
path = "ex2data2.txt"
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

print(data2.head())

po = data2[data2['Accepted'].isin([1])]
ne = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(po['Test 1'], po['Test 2'], c='b', marker='o')
ax.scatter(ne['Test 1'], ne['Test 2'], c='r', marker='x')

ax.legend
# plt.show()

# 多项式

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)


data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

print(data2.head())


