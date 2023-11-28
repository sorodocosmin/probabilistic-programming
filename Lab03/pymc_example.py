import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

def main():

    model = pm.Model()

    with model:
        urgent = pm.Bernoulli('urgent', 0.05)
        reducere = pm.Bernoulli('reducere', 0.2)
        cumpara_produs = pm.Deterministic('cumpara_produs',
                                          pm.math.switch(reducere,
                                                         pm.math.switch(urgent, 1, 0.5),
                                                         pm.math.switch(urgent, 0.8, 0.2)))

        cumpara = pm.Bernoulli('cumpara', p=cumpara_produs, observed=1)

    with model:
        trace = pm.sample(20_000)

    df = trace.to_dataframe(trace)

    p_urgent = df[(df['urgent'] == 1)].shape[0] / df.shape[0]
    print(p_urgent)

    az.plot_posterior(trace)
    plt.show()


if __name__ == '__main__':
    main()