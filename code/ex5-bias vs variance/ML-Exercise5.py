print("123")
import numpy as np
import scipy.io as sio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt


def load_data():
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])


X, y, Xval, yval, Xtest, ytest = load_data()

print(X.shape)
print(y.shape)
print(Xval.shape)
print(yval.shape)
print(Xtest.shape)
print(ytest.shape)

df = pd.DataFrame({'water_level': X, 'flow': y})

# sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)
sns.lmplot(data=df, x="water_level", y="flow", fit_reg=False, )

# plt.show()

X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

print(X.shape)
print(Xval.shape)
print(Xtest.shape)


# 没看懂
def cost(theta, X, y):
    m = X.shape[0]

    # 我还是没看懂怎么回事。
    inner = X @ theta - y  # R(m*1)

    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost


theta = np.ones(X.shape[1])
costValue = cost(theta, X, y)

print(costValue)


# 我没看懂额, 核心，我没看懂。但是
def gradient(theta, X, y):
    m = X.shape[0]

    # 为什么要用 X的 转制乘法, 我没看懂
    inner = X.T @ (X @ theta - y)

    # inner 是个两纬数组，代表什么意思。one degree

    return inner / m


gradientValue = gradient(theta, X, y)

print(gradientValue)


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0

    regularized_term = (1 / m) * regularized_term

    return gradient(theta, X, y) + regularized_term


gradientValue2 = regularized_gradient(theta, X, y)
print("regularized_gradient:")
print(gradientValue2)


def linear_regression_np(X, y, l=1):
    """
    :param X: feature matrix, (m, n+1) # with intercept x0=1
    :param y: target vector, (m,)
    :param l: lambda constant for regularization
    :return:
    """
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method="TNC",
                       jac=regularized_gradient,
                       options={'disp': True}
                       )

    return res


def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()

    return cost(theta, X, y) + regularized_term


theta = np.ones(X.shape[1])

costValue1 = regularized_cost(theta, X, y, l=0)
print(costValue1)

final_theta = linear_regression_np(X, y, l=0).get('x')
print(final_theta)

b = final_theta[0]
m = final_theta[1]

plt.scatter(X[:, 1], y, label="Tranning data")
plt.plot(X[:, 1], X[:, 1] * m + b, label="Prediction")
plt.legend(loc=2)
# plt.show()
## 神奇

### 完全没搞懂，怎么来的。


training_cost, cv_cost = [], []

m = X.shape[0]
for i in range(1, m + 1):
    print('i={}'.format(i))
    res = linear_regression_np(X[:i, :], y[:i], l=0)
    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)

    print('tc={}, cv={}'.format(tc, cv))
    training_cost.append(tc)
    cv_cost.append(cv)

print(training_cost)
print(cv_cost)

plt.close()


# plt.plot(np.arange(1, m + 1), training_cost, label="training cost")
# plt.plot(np.arange(1, m + 1), cv_cost, label="cv cot")
# plt.legend(loc=1)
# plt.show()


def prepare_poly_data(*args, power):
    """
    :param args: keep feeding in X, Xval, or Xtest will return in the same order
    :param power:
    :return:
    """

    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        ndarr = normalize_feature(df).values

        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]


def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.values if as_ndarray else df


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


X, y, Xval, yval, Xtest, ytest = load_data()

poly_features_value = poly_features(X, power=3)
print(poly_features_value)

X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)
print(X_poly)
print(Xval_poly)
print(Xtest_poly)


# 完全看不懂
def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        res = linear_regression_np(X[:i, :], y[:i], l=l)

        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label="training cost")
    plt.plot(np.arange(1, m + 1), cv_cost, label="cv cost")

    plt.legend(loc=1)


plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
plt.show()

