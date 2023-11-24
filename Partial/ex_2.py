import arviz as az
import pymc as pm
from scipy import stats
import matplotlib.pyplot as plt



def main():

    # Generăm o distribuție normală a timpilor de așteptare, cu o medie de 10 și o deviație standard de 2, pentru 100 de eșantioane.
    miu = 10
    sigma = 2.5
    # ex 2.1 -- generarea a 200 timp medii de asteptare
    timp_mediu_asteptare = stats.norm.rvs(miu, sigma, size=200)
    # print(timp_mediu_asteptare)

    model = pm.Model()

    with model:
        # pentru acest model, incercam sa definim noi distributii pemtru mu si sigma

        # 'generam' pentru miu o distributie normala
        miu_d = pm.Normal('miu', mu=miu) # pentru a defini o distribuție normală în cadrul unui model Bayesian
        # pentru deviatia standard, generam o distributie HalfNormala (pt val pozitive)
        sigma_d = pm.HalfNormal('sigma', sigma=sigma)

        # acum, tinand cont de distributiile definite mai sus, observam timpul mediu de asteptare
        # in acest caz, atunci cand facem "trace" modelul v-a incerca sa gaseasca cele mai bune valori pentru miu si sigma
        timpi_medii_observati = pm.Normal('timp_obs', mu=miu_d, sigma=sigma_d, observed=timp_mediu_asteptare)

    with model:
        trace = pm.sample(1000, tune=1000)

    az.plot_posterior(trace, var_names=['sigma'])
    #pentru a vizualiza distribuția posterioară a parametrilor modelului, in cazul nostru sigma

    plt.title('Distribuția a posteriori pentru sigma')
    plt.xlabel('Sigma')
    plt.show()
    # din grafic putem observa ca valoarea lui sigma este aproximativ sigma(pe care l-am definit la inceput)
    # deci, da, acesta corespunde asteparilor noastre

if __name__ == "__main__":
    main()