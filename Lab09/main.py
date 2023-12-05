import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def read_data():    # a
    file_path = 'Admission.csv'
    df = pd.read_csv(file_path)

    admission = df['Admission'].values.astype(int)
    gre = df['GRE'].values.astype(float)
    gpa = df['GPA'].values.astype(float)
    return np.array(admission), np.array(gre), np.array(gpa)


def main():
    # admission, gre, gpa = read_data()

    file_path = 'Admission.csv'
    df = pd.read_csv(file_path)
    admission = df['Admission'].values.astype(int)
    attributes = ['GRE', 'GPA']
    x_1 = df[attributes].values.astype(float)


    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1, shape=len(attributes))

        miu = alpha + pm.math.dot(x_1, beta)
        #theta = pm.Deterministic('theta', pm.math.sigmoid(miu))
        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-miu)))
        bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:, 0])

        admission_observed = pm.Bernoulli('admission_observed', p=theta, observed=admission)

        idata = pm.sample(500, return_inferencedata=True)

    idx = np.argsort(x_1[:, 0])
    bd = idata.posterior['bd'].mean(('chain', 'draw'))[idx]
    plt.scatter(x_1[:, 0], x_1[:, 1], c=[f"C{x}" for x in admission])
    plt.plot(x_1[:, 0][idx], bd, color='k')
    az.plot_hdi(x_1[:, 0], idata.posterior['bd'], color='k')
    plt.xlabel(attributes[0])
    plt.ylabel(attributes[1])

    plt.show()


if __name__ == "__main__":
    main()