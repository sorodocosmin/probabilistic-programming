import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


def main():
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    # ex_1(x_1, y_1)
    ex_2(x_1, y_1, dummy_data)


def ex_2(x_1, y_1, dummy_data):

    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    # Calculate the mean and covariance of the original points
    mean = np.mean(dummy_data, axis=0)
    cov = np.cov(dummy_data, rowvar=False)

    # pentru a avea exact 500 de puncte
    # incercam sa generam noile puncte in jurul punctelor originale
    x_extended, y_extended = np.random.multivariate_normal(mean, cov, 500 - len(x_1)).T

    # Combine the original and extended points
    x_combined = np.concatenate((x_1, x_extended))
    y_combined = np.concatenate((y_1, y_extended))


    order = 5
    x_1p = np.vstack([x_combined**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_combined - y_combined.mean()) / y_combined.std()

    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    #ex_1_a(x_1s, y_1s, order)
    ex_1_b(x_1s, y_1s, order)



def ex_1(x_1, y_1):
    order = 5
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    ex_1_a(x_1s, y_1s, order)
    ex_1_b(x_1s, y_1s, order)


def ex_1_a(x_1s, y_1s, order):
    #linear model
    with pm.Model() as model_linear:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10)
        eps = pm.HalfNormal('eps', 5)
        miu = alpha + beta * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
        idata_linear = pm.sample(10, return_inferencedata=True)

    with pm.Model() as model_order:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        eps = pm.HalfNormal('eps', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
        idata_order = pm.sample(10, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 500)
    # plot linear model
    alpha_linear_post = idata_linear.posterior['alpha'].mean(("chain", "draw")).values
    beta_linear_post = idata_linear.posterior['beta'].mean(("chain", "draw")).values
    y_linear_post = alpha_linear_post + beta_linear_post * x_new
    plt.plot(x_new, y_linear_post, 'C1', label='linear model')

    # plot order model
    plot_order_model(x_1s, y_1s, idata_order, f'model order {order}', 'C2')

    plt.legend()
    plt.show()


def ex_1_b(x_1s, y_1s, order):
    # using sd=100
    with pm.Model() as model_order_sd_100:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
        eps = pm.HalfNormal('eps', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
        idata_order_sd_100 = pm.sample(10, return_inferencedata=True)

    # using sd=np.array
    arr = np.array([10, 0.1, 0.1, 0.1, 0.1])
    with pm.Model() as model_order_sd_array:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=arr, shape=order)
        eps = pm.HalfNormal('eps', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
        idata_order_sd_array = pm.sample(10, return_inferencedata=True)

    # plot order model sd=100
    plot_order_model(x_1s, y_1s, idata_order_sd_100, f'model order {order} - sd=100', 'C1')

    # plot order model sd=np.array
    plot_order_model(x_1s, y_1s, idata_order_sd_array, f'model order {order} - sd=np.array()', 'C2')

    plt.legend()
    plt.show()


def plot_order_model(x_1s, y_1s, idata, label, c):
    alpha_order_post = idata.posterior['alpha'].mean(("chain", "draw")).values
    beta_order_post = idata.posterior['beta'].mean(("chain", "draw")).values

    index = np.argsort(x_1s[0])
    y_order_post = alpha_order_post + np.dot(beta_order_post, x_1s)

    plt.plot(x_1s[0][index], y_order_post[index], c, label=label)

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')


if __name__ == '__main__':
    main()
