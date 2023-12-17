from ex_1_and_2 import plot_order_model
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az



def main():
    dummy_data = np.loadtxt('dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    x_1p = np.vstack([x_1**i for i in range(1, 3+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    ex_3(x_1s, y_1s)


def ex_3(x_1s, y_1s):
    #linear model
    with pm.Model() as model_linear:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10)
        eps = pm.HalfNormal('eps', 5)
        miu = alpha + beta * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
        idata_linear = pm.sample(100, return_inferencedata=True)

    with pm.Model() as model_quadratic:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        eps = pm.HalfNormal('eps', 5)
        miu = alpha + pm.math.dot(beta, x_1s[:2])
        y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
        idata_quadratic = pm.sample(100, return_inferencedata=True)

    with pm.Model() as model_cubic:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=3)
        eps = pm.HalfNormal('eps', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=eps, observed=y_1s)
        idata_cubic = pm.sample(100, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    # plot linear model

    alpha_linear_post = idata_linear.posterior['alpha'].mean(("chain", "draw")).values
    beta_linear_post = idata_linear.posterior['beta'].mean(("chain", "draw")).values
    y_linear_post = alpha_linear_post + beta_linear_post * x_new
    plt.plot(x_new, y_linear_post, 'C1', label='linear model')

    # plot model quadratic
    plot_order_model(x_1s[:2], y_1s, idata_quadratic, f'model order {2}', 'C2')

    # plot model cubic
    plot_order_model(x_1s, y_1s, idata_cubic, f'model order {3}', 'C3')

    plt.legend()
    plt.show()

    # comparare modele
    waic_linear, loo_linear = compute(model_linear, idata_linear, 'linear')
    waic_quadratic, loo_quadratic = compute(model_quadratic, idata_quadratic, 'quadratic')
    waic_cubic, loo_cubic = compute(model_cubic, idata_cubic, 'cubic')

    # sau asa cum este si in curs:

    cmp_df = az.compare({'model_linear': idata_linear, 'model_quadratic': idata_quadratic, 'model_cubic': idata_cubic},

                        method='BB-pseudo-BMA', ic="waic", scale="deviance")
    print(cmp_df)
    az.plot_compare(cmp_df)

    cmp_df = az.compare({'model_linear': idata_linear, 'model_quadratic': idata_quadratic, 'model_cubic': idata_cubic},

                        method='BB-pseudo-BMA', ic="loo", scale="deviance")
    print(cmp_df)
    az.plot_compare(cmp_df)
    plt.show()


def compute(model, idata, label):

    pm.compute_log_likelihood(idata, model=model)
    waic = az.waic(idata, scale='deviance')
    loo = az.loo(idata, scale='deviance')

    print(f'Model {label}')
    print(f'WAIC: {waic}')
    print(f'LOO: {loo}')

    return waic, loo


if __name__ == '__main__':
    main()