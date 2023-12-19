import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import scipy.stats as stats


# Ex1
def main():
    nr_distributions = 3  # 3 distributii Gaussiene
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)

    # media si deviatia standard a fiecarei distributii
    means = [5, 0, -5]
    std_devs = [2, 2, 2]


    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

    datas = np.array(mix)
    az.plot_kde(datas)
    plt.show()


    # Ex2
    nr_components = [2, 3, 4]

    models = []
    idatas = []

    for cluster in nr_components:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                                mu=np.linspace(datas.min(), datas.max(), cluster),
                                sigma=10, shape=cluster,
                                transform=pm.distributions.transforms.ordered)

            sd = pm.HalfNormal('sd', sigma=10)
            y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=datas)
            idata = pm.sample(100, tune=100, target_accept=0.9, random_seed=112, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)

    _, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = np.ravel(ax)
    x = np.linspace(datas.min(), datas.max(), 200)

    for idx, idata_x in enumerate(idatas):
        posterior_x = idata_x.posterior.stack(samples=("chain", "draw"))
        x_ = np.array([x] * nr_components[idx]).T
        for i in range(50):
            i_ = np.random.randint(0, posterior_x.samples.size)
            means_y = posterior_x['means'][:, i_]
            p_y = posterior_x['p'][:, i_]
            sd = posterior_x['sd'][i_]
            dist = stats.norm(means_y, sd)
            ax[idx].plot(x, np.sum(dist.pdf(x_) * p_y.values, 1), 'C0', alpha=0.1)
        means_y = posterior_x['means'].mean("samples")
        p_y = posterior_x['p'].mean("samples")
        sd = posterior_x['sd'].mean()
        dist = stats.norm(means_y, sd)
        ax[idx].plot(x, np.sum(dist.pdf(x_) * p_y.values, 1), 'C0', lw=2)
        ax[idx].plot(x, dist.pdf(x_) * p_y.values, 'k--', alpha=0.7)
        az.plot_kde(datas, plot_kwargs={'linewidth': 2, 'color': 'k'}, ax=ax[idx])
        ax[idx].set_title(f'K = {nr_components[idx]}')
        ax[idx].set_yticks([])
        ax[idx].set_xlabel('x')

    plt.show()

    # Ex3
    # comparare cu criteriul waic
    [pm.compute_log_likelihood(idatas[i], model=models[i]) for i in range(3)]
    comp = az.compare(dict(zip([str(c) for c in nr_components], idatas)),
                      method='BB-pseudo-BMA', ic="waic", scale="deviance")

    print(comp)
    az.plot_compare(comp)
    plt.show()

    # comparare cu criteriul leave one out (loo)
    comp = az.compare(dict(zip([str(c) for c in nr_components], idatas)),
                      method='BB-pseudo-BMA', ic="loo", scale="deviance")

    print(comp)
    az.plot_compare(comp)
    plt.show()

    # Se poate observa ca pentru K=2 nu este prea potrivita
    # K=3 sau 4 fiind mai bune
    # De asemenea, se observa ca minimul este obtinut pentru k=4, insa intre k=4 si
    # k=3 nu este o diferenta foarte mare.
    # De obicei, modelele cu elpd_waic mare si p_waic(numarul efectiv de parametri) mici sunt preferate,
    # Deci, in acest caz k=3 este mai potrivit decat k=4.


if __name__ == '__main__':
    main()
