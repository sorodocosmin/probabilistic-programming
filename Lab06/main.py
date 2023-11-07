import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

possible_Y = [0, 5, 10]
possible_prob_clients = [0.2, 0.5]

nr_clients = 10

fig, axes = plt.subplots(len(possible_Y), len(possible_prob_clients), figsize=(12, 8))


for i, y in enumerate(possible_Y):
    for j, prob_client in enumerate(possible_prob_clients):
        with pm.Model() as model:
            n = pm.Poisson("n", mu=nr_clients)
            name_distribution = f"Y={y}, prob_client={prob_client}"

            distribution_binomial = pm.Binomial(name_distribution, n=n, p=prob_client, observed=y)
            trace = pm.sample(1000, tune=1000, cores=1)
            az.plot_posterior(trace, var_names=["n"], round_to=2, point_estimate="mean",
                              ax=axes[i, j])
            axes[i, j].set_title(f"Y={y}, th={prob_client}")

plt.tight_layout()

plt.show()

