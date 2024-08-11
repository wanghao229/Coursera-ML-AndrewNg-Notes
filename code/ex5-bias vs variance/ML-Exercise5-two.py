import scipy.io as sio
import numpy as np
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

# sns.lmplot(data=df, x="water_level", y="flow", fit_reg=False)
# plt.show()

X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

print(X.shape)
print(Xval.shape)
print(Xtest.shape)


def cost(theta, X, y):
    m = X.shape[0]

    inner = X @ theta - y
    square_sum = inner.T @ inner

    cost = square_sum / (2 * m)

    return cost


theta = np.ones(X.shape[1])

print("\n")
print("cost(theta, X, y)")
print(cost(theta, X, y))


### 我虽然不知道原理，但是我能一步步验证
def gradient(theta, X, y):
    m = X.shape[0]

    ### 为什么这么写
    inner = X.T @ (X @ theta - y)

    return inner / m


print("\n")
print("gradient(theta, X, y)")
print(gradient(theta, X, y))


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = theta.copy()
    regularized_term[0] = 0

    regularized_term = (l / m) * regularized_term

    return gradient(theta, X, y) + regularized_term


print("\n")
print("regularized_gradient(theta, X, y)")
print(regularized_gradient(theta, X, y))


### 正则化代价， 没看懂
def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()

    return cost(theta, X, y) + regularized_term


def linear_regression_np(X, y, l=1):
    theta = np.ones(X.shape[1])

    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True}

                       )

    return res


theta = np.ones(X.shape[0])
final_theta = linear_regression_np(X, y, l=0).get('x')

b = final_theta[0]
m = final_theta[1]

print("\n")
print("final_theta")
print(final_theta)

# plt.scatter(X[:, 1], y, label="Training data")
# plt.plot(X[:, 1], X[:, 1] * m + b, label="Prediction")
# plt.legend(loc=2)
# plt.show()


training_cost, cv_cost = [], []

m = X.shape[0]
for i in range(1, m + 1):
    res = linear_regression_np(X[:i, :], y[:i], l=0)

    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)

    training_cost.append(tc)
    cv_cost.append(cv)

print("\n")
print("training_cost")
print("cv_cost")
print(training_cost)
print(cv_cost)


# plt.plot(np.arange(1, m + 1), training_cost, label="training cost")
# plt.plot(np.arange(1, m + 1), cv_cost, label="cv cost")
#
# plt.legend(loc=1)
# plt.show()

def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())


def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}

    df = pd.DataFrame(data)

    return df.values if as_ndarray else df


def prepare_poly_data(*args, power):
    def prepare(x):
        df = poly_features(x, power=power)

        ndarr = normalize_feature(df).values

        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]


X, y, Xval, yval, Xtest, ytest = load_data()

print("\n")
print("poly_features(X, power=3)")
print(poly_features(X, power=3))

X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)
print("\n")
print("X_poly[:3, :]")
print(X_poly[:3, :])


def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []

    m = X.shape[0]

    for i in range(1, 1 + m):
        res = linear_regression_np(X[:i, :], y[:i], l=l)

        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)


### 有多项式后是什么意思，我没看懂。
plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
plt.show()

l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []

for l in l_candidate:
    res = linear_regression_np(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)

# plt.plot(l_candidate, training_cost, label='training')
# plt.plot(l_candidate, cv_cost, label='cross validation')
# plt.legend(loc=2)
#
# plt.xlabel('lambda')
#
# plt.ylabel('cost')
# plt.show()


print("\n")
print("l_candidate[np.argmin(cv_cost)]")
print(l_candidate[np.argmin(cv_cost)])

for l in l_candidate:
    theta = linear_regression_np(X_poly, y, l).x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))

l = 0.3

theta = linear_regression_np(X_poly, y, l).x
print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))

### result
### l =3 is best
