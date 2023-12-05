import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def show_gre_gpa(idata, gre, gpa):
    alpha = idata.posterior['alpha'].mean(('chain', 'draw'))
    beta = idata.posterior['beta'].mean(('chain', 'draw'))

    probabilities = 1 / (1 + np.exp(-(alpha + gre * beta[0] + gpa * beta[1])))

    # intervalul de 90% de probabilitati
    hdi_probabilities = az.hdi(probabilities, hdi_prob=0.9)

    hdi_indices = np.where((probabilities >= hdi_probabilities[0]) & (probabilities <= hdi_probabilities[1]))
    hdi_probabilities = probabilities[hdi_indices]

    hdi_probabilities.sort()

    plt.figure()
    plt.plot(hdi_probabilities)
    plt.title(f'GRE = {gre}, GPA = {gpa}')
    plt.show()


def main():
    file_path = 'Admission.csv'
    df = pd.read_csv(file_path)
    admission = df['Admission'].values.astype(int)
    attributes = ['GRE', 'GPA']
    x_1 = df[attributes].values.astype(float)
    # print(f"X : \n {x_1}")

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=len(attributes))
        miu = alpha + pm.math.dot(x_1, beta)
        #theta = pm.Deterministic('theta', pm.math.sigmoid(miu))
        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-miu)))
        bd = pm.Deterministic('bd', -alpha/beta[0] - beta[1]/beta[0] * x_1[:, 1])

        admission_observed = pm.Bernoulli('admission_observed', p=theta, observed=admission)

        idata = pm.sample(1000, return_inferencedata=True)



    idx = np.argsort(x_1[:, 1])
    bd = idata.posterior['bd'].mean(('chain', 'draw'))[idx]

    print(bd.mean())

    plt.scatter(x_1[:, 1], x_1[:, 0], c=[f"C{x}" for x in admission])
    plt.plot(x_1[:, 1][idx], bd, color='k')
    az.plot_hdi(x_1[:, 1], idata.posterior['bd'], color='k', hdi_prob=0.94)  # intervalul de HDI de 94%
    plt.xlabel(attributes[1])
    plt.ylabel(attributes[0])
    plt.show()

    # ex 3 + 4
    show_gre_gpa(idata, gre=550, gpa=3.5)
    show_gre_gpa(idata, gre=500, gpa=3.2)


if __name__ == "__main__":
    main()
