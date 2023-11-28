import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az


# pentru x2 vom folosi log, a.i. sa nu avem valori prea mari
def read_data():    # a
    file_path = 'Prices.csv'
    df = pd.read_csv(file_path)
    # df = df[df['horsepower'] != '?']  nu avem nevoie de preprocesare

    speed = df['Speed'].values.astype(float)
    size_hard_drive = df['HardDrive'].values.astype(float)
    price = df['Price'].values.astype(float)

    return np.array(speed), np.log(size_hard_drive), np.array(price)


def main():
    speed, log_size_hard_drive, price = read_data()
    # print(f"price : \n {price}")
    # print(f"size hard drive : \n {log_size_hard_drive}")

    # definim modelul de regresit cu pymc
    with pm.Model() as model_regression:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=1)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=1)

        eps = pm.HalfCauchy('eps', 5)
        miu = pm.Deterministic('miu', alpha + beta_1 * speed + beta_2 * log_size_hard_drive)

        # Modificarea aici: variabila observată trebuie să fie 'price', nu 'price_pred'
        price_observed = pm.Normal('price_observed', mu=miu, sigma=eps, observed=price)
        idata = pm.sample(800, tune=800, return_inferencedata=True)

    az.plot_trace(idata, var_names=['beta_1', 'beta_2'])
    plt.show()

    hdi_95_beta1 = pm.stats.hdi(idata.posterior["beta_1"], hdi_prob=0.95)
    hdi_95_beta2 = pm.stats.hdi(idata.posterior["beta_2"], hdi_prob=0.95)

    print(f"Intervalul de Încredere de 95% pentru beta_1: {hdi_95_beta1}")
    print(f"Intervalul de Încredere de 95% pentru beta_2: {hdi_95_beta2}")



if __name__ == "__main__":
    main()
