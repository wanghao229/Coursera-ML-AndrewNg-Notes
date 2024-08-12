from scipy.io import loadmat
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

print("ML-Exercise6")

raw_data = loadmat('data/ex6data1.mat')

print("\nraw_data")
print(raw_data)

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])

print("\ndata")
print(data)

data['y'] = raw_data['y']
print("\ndata")
print(data)

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

# print("positive")
# print(positive)

# print("negative")
# print(negative)

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')

# ax.legend()
# plt.show()

from sklearn import svm

svc = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
print(svc)

svc.fit(data[['X1', 'X2']], data['y'])
print("\nsvc.score(data[['X1', 'X2']], data['y'])")
### 这个score 是什么意思，没看懂
print(svc.score(data[['X1', 'X2']], data['y']))

data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])

print("\ndata['SVM 1 Confidence']")
print(data)

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')

ax.set_title('SVM(C=1) Decision Confidence')


# plt.show()


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))


x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])

sigma = 2

print("\ngaussian_kernel(x1, x2, sigma)")
print(gaussian_kernel(x1, x2, sigma))

# raw_data2 = loadmat('data/ex6data2.mat')
raw_data2 = loadmat('data/ex6data2.mat')
# print(raw_data2)
data = pd.DataFrame(raw_data2['X'], columns=['X1', 'X2'])
data['y'] = raw_data2['y']

print("\ndata2")
print(data)

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label="Positive")
ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label="Negative")

ax.legend()
# plt.show()

svc = svm.SVC(C=100, gamma=10, probability=True)
print(svc)

svc.fit(data[['X1', 'X2']], data['y'])
print("\nsvc.score(data[['X1', 'X2']], data['y'])")
print(svc.score(data[['X1', 'X2']], data['y']))

data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:, 0]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds')
# plt.show()

raw_data = loadmat('data/ex6data3.mat')

X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y'].ravel()
yval = raw_data['yval'].ravel()

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)

        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print("\nbest_score, best_params")
print(best_score, best_params)

spam_train = loadmat('data/spamTrain.mat')
spam_test = loadmat('data/spamTest.mat')

print("spam_train")
print(spam_train)

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

print("X.shape, y.shape, Xtest.shape, ytest.shape")
print(X.shape, y.shape, Xtest.shape, ytest.shape)

svc = svm.SVC()
svc.fit(X, y)

print('Training accuracy = {0}%'
      .format(np.round(svc.score(X, y) * 100, 2)))


print('Test accuracy = {0}%'
      .format(np.round(svc.score(Xtest, ytest) * 100, 2)))
