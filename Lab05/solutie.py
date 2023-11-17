import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats

count_data = np.loadtxt("trafic.csv", delimiter=',', dtype=int)
n_count_data = len(count_data)
with pm.Model() as model:
    alpha = 1.0 / count_data[:, 1].mean()

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    lambda_4 = pm.Exponential("lambda_4", alpha)
    lambda_5 = pm.Exponential("lambda_5", alpha)

    interval1 = pm.Normal("interval1", 60 * 3)
    interval2 = pm.Normal("interval2", 60 * 12)
    interval3 = pm.Normal("interval3", 60 * 15)
    interval4 = pm.Normal("interval4", 60 * 20)

    tau1 = pm.DiscreteUniform("tau1", lower=1, upper=interval1)
    tau2 = pm.DiscreteUniform("tau2", lower=tau1, upper=interval2)
    tau3 = pm.DiscreteUniform("tau3", lower=tau2, upper=interval3)
    tau4 = pm.DiscreteUniform("tau4", lower=tau3, upper=interval4)

    idx = np.arange(n_count_data)
    lmbd1 = pm.math.switch(tau1 > idx, lambda_1, lambda_2)
    lmbd2 = pm.math.switch(tau2 > idx, lmbd1, lambda_3)
    lmbd3 = pm.math.switch(tau3 > idx, lmbd2, lambda_4)
    lmbd4 = pm.math.switch(tau4 > idx, lmbd3, lambda_5)
    observation = pm.Poisson("obs", lmbd4, observed=count_data[:, 1])
    trace = pm.sample(10, cores=1)
    az.plot_posterior(trace)
    plt.show()

client_nr = stats.poisson.rvs(20, size=1)

command_nr = stats.norm.rvs(loc=2, scale=0.5, size=20)


def calc(guess):
    rand_gen = stats.expon.rvs(loc=guess, size=20)
    count = 0
    for element in rand_gen:
        if element < 15:
            count = count + 1
    if count / 20 * 100 > 95:
        return calc(guess + 1)
    else:
        return guess


alpha = calc(1)
print(alpha)

meanlist = []
for i in range(100):
    cook_nr = stats.expon.rvs(loc=3, size=20).mean()
    meanlist.append(copy.deepcopy(cook_nr))

with pm.Model() as model:
    alpha = 3
    nr_clienti = pm.Poisson("nr_clienti", mu=20)
    timpPlasarePLata = pm.Normal("timpPlasarePlata", mu=2, sigma=0.5)
    timpPregatire = pm.Exponential("timpPregatire", mu=alpha)
    observation = pm.Poisson("obs", mu=timpPregatire, observed=meanlist)


    with model:
        trace = pm.sample(1000, cores=1)
        az.plot_posterior(trace)
        plt.show()