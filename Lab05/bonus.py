import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az
from scipy.stats import poisson, norm, expon


nr_simulari = 10_000
lambda_client = 20
media_timp_comanda = 2
deviatia_standard_comanda = 0.5
alpha = 3

timpi_medii = []
for _ in range(100):
    timp_pregatire_comanda_expon = expon.rvs(0, alpha, size=nr_simulari)
    timp_plasare_si_plata_norm = norm.rvs(media_timp_comanda, deviatia_standard_comanda, size=nr_simulari)

    timp_servire = timp_pregatire_comanda_expon + timp_plasare_si_plata_norm

    timpi_medii.append(np.mean(timp_servire))

print("Timpi medii: ", timpi_medii)

model = pm.Model()

with model:
    preg_comanda_expon = pm.Exponential("preg_comanda_expon", lam=1/alpha)

    timp_asteptare_obs = pm.Normal("timp_asteptare_obs", mu=media_timp_comanda+alpha, observed=timpi_medii)

    trace = pm.sample(1000, tune=1000, cores=1)

az.plot_posterior(trace)
plt.show()

