import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # ex - 1
    centered_eight = az.load_arviz_data('centered_eight')
    non_centered_eight = az.load_arviz_data('non_centered_eight')

    num_chains_centered = centered_eight.posterior.chain.size
    total_samples_centered = centered_eight.posterior.draw.size
    print(f"Numarul de lanturi pentu modelul centrat : {num_chains_centered}")
    print(f"Numarul de esantioane pentru modelul centrat : {total_samples_centered}")

    num_chains_non_centered = non_centered_eight.posterior.chain.size
    total_samples_non_centered = non_centered_eight.posterior.draw.size
    print(f"Numarul de lanturi pentru modelul necentrat este : {num_chains_non_centered}")
    print(f"Numarul de esantioane pentru modelul necentrat este : {total_samples_non_centered}")

    az.plot_posterior(centered_eight)

    az.plot_posterior(non_centered_eight)

    plt.show()

    # ex - 2

    rhat_centered = az.rhat(centered_eight, var_names=["mu", "tau"])
    rhat_non_centered = az.rhat(non_centered_eight, var_names=["mu", "tau"])

    print("Rhat pentru modelul centrat (mu și tau):", rhat_centered)
    print("Rhat pentru modelul necentrat (mu și tau):", rhat_non_centered)
    # se poate observa ca valoarea obtinuta pentru Rhat, atat pentru tau, cat si pentru mu, este mai mic decat 1.1,
    # (care in practica este o valoare buna), deci putem spune ca modelele sunt convergente
    # Pentru primul model(cel centrat) pentru parametrii mu si tau, Rhat este 1.02, 1.062, iar pentru
    # al doilea model(cel necentrat) pentru parametrii mu si tau, Rhat este 1.003, 1.003,
    # (Obs1)Am m-ai putea spune ca primul model converge mai lent decat al 2-lea

    # sau o alta metoda pt afisarea Rhat pentru fiecare model
    print(f"Rhat pentru cele 2 modele, in functie de parametrul mu")
    summaries = pd.concat([az.summary(centered_eight, var_names=['mu']), az.summary(non_centered_eight, var_names=['mu'])])
    summaries.index = ['centered', 'non_centered']
    print(summaries)
    print(f"Rhat pentru cele 2 modele, in functie de parametrul tau")
    summaries = pd.concat([az.summary(centered_eight, var_names=['tau']), az.summary(non_centered_eight, var_names=['tau'])])
    summaries.index = ['centered', 'non_centered']
    print(summaries)
    print("Versiunea ArviZ instalată:", az.__version__)

    # Autocorelatia
    autocorr_centered_mu = az.autocorr(centered_eight.posterior["mu"].values)
    autocorr_centered_tau = az.autocorr(centered_eight.posterior["tau"].values)
    autocorr_non_centered_mu = az.autocorr(non_centered_eight.posterior["mu"].values)
    autocorr_non_centered_tau = az.autocorr(non_centered_eight.posterior["tau"].values)

    print(f"Autocorelatia pentru modelul centrat, in functie de parametrul mu : {np.mean(autocorr_centered_mu)}")
    print(f"Autocorelatia pentru modelul centrat, in functie de parametrul tau : {np.mean(autocorr_centered_tau)}")
    print(f"Autocorelatia pentru modelul necentrat, in functie de parametrul mu : {np.mean(autocorr_non_centered_mu)}")
    print(f"Autocorelatia pentru modelul necentrat, in functie de parametrul tau : {np.mean(autocorr_non_centered_tau)}")
    # se poate observa ca pentru amble modele autocorelatia este aproape de 0
    # Plotarea autocorelatiei

    az.plot_autocorr(centered_eight, var_names=["mu", "tau"], combined=True, figsize=(10, 5))
    az.plot_autocorr(non_centered_eight, var_names=["mu", "tau"], combined=True, figsize=(10, 5))
    # Similar cu (Obs1), si din compararea acestor modele dupa criteriul autocorelatiei, se poate observa ca pentru
    # al 2-lea model autocorelatia este mai scazuta --> convergenta mai rapida

    plt.show()



    # ex - 3

    divergences_centered = centered_eight.sample_stats["diverging"].sum()
    divergences_non_centered = non_centered_eight.sample_stats["diverging"].sum()

    print()
    print(f"Numărul de divergențe pentru modelul centrat: {divergences_centered.values}")
    print(f"Numărul de divergențe pentru modelul non-centrat: {divergences_non_centered.values}")

    az.plot_pair(centered_eight, var_names=["mu", "tau"], divergences=True)
    plt.suptitle("Model centrat")
    plt.show()

    az.plot_pair(non_centered_eight, var_names=["mu", "tau"], divergences=True)
    plt.suptitle("Model necentrat")
    plt.show()

    # plot paralel
    az.plot_parallel(centered_eight, var_names=["mu", "tau"])
    plt.suptitle("Plot Parallel - Model centrat")
    plt.show()

    az.plot_parallel(non_centered_eight, var_names=["mu", "tau"])
    plt.suptitle("Plot Parallel - Model necentrat")
    plt.show()


if __name__ == '__main__':
    main()
