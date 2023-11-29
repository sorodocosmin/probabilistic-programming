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
    premium = (df['Premium'].values == "yes").astype(int)
    return np.array(speed), np.log(size_hard_drive), np.array(price), np.array(premium)


def main():
    speed, log_size_hard_drive, price, premium = read_data()  # premium folosit pentru bonus
    # print(f"price : \n {price}")
    # print(f"size hard drive : \n {log_size_hard_drive}")

    # definim modelul de regresit cu pymc

    # POINT 1
    with pm.Model() as model_regression:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=1)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=1)

        eps = pm.HalfCauchy('eps', 5)
        miu = pm.Deterministic('miu', alpha + beta_1 * speed + beta_2 * log_size_hard_drive)

        price_observed = pm.Normal('price_observed', mu=miu, sigma=eps, observed=price)
        idata = pm.sample(5000, return_inferencedata=True)


    # az.plot_trace(idata, var_names=['beta_1', 'beta_2'])
    # plt.show()

    # POINT 2 + 3

    pm.plot_posterior(idata, var_names=['beta_1', 'beta_2'], hdi_prob=0.95)
    plt.show()

    hdi_beta1 = az.hdi(idata['posterior']['beta_1'], hdi_prob=0.95)
    hdi_beta2 = az.hdi(idata['posterior']['beta_2'], hdi_prob=0.95)

    print(f"\n95% HDI pentru beta1: {hdi_beta1}\n")
    # Intervalul obtinut : [13.08, 17.62]
    # Tinand cont ca acest interval nu contine 0, putem spune ca
    # Frecventa Procesorului are un impact seminifcativ asupra pretului,
    # adica cu cat frecventa procesorului este mai mare, cu atat pretul este mai mare
    # Deci DA, frecventa procesorului ESTE un predictor util pentru pretul de vanzare


    print(f"\n95% HDI pentru beta2: {hdi_beta2}\n")
    # Intervalul obtinut : [-0.1215, 3.841]
    # Acest interval il contine pe 0, deci puntem spune ca
    # in unele cazuri coeficientul pentru marimea hard_disk-ului
    # va fi 0, deci valoarea dim_hardisk nu va infuemnta pretul de vanzare
    # Deci, marimea hard diskului NU ESTE un predictor foarte utill


    # POINT 4


    # BONUS

    # trebuie sa introducem in model si variabila pentru Premium

    with pm.Model() as model_regression_2:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=1)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=1)
        beta_3 = pm.Normal('beta_3', mu=0, sigma=1)

        eps = pm.HalfCauchy('eps', 5)
        miu = pm.Deterministic('miu', alpha + beta_1 * speed + beta_2 * log_size_hard_drive + beta_3 * premium)

        price_observed = pm.Normal('price_observed', mu=miu, sigma=eps, observed=price)
        idata = pm.sample(1_000, tune=1_000, return_inferencedata=True)

    # ne uitam la intervalul obtinut pentru coeficientul lui premium ( beta_3 )
    hdi_premium = az.hdi(idata['posterior']['beta_3'], hdi_prob=0.95)
    print(f"95% HDI pentru beta3 (premium): {hdi_premium}")
    pm.plot_posterior(idata, var_names=['beta_3'], hdi_prob=0.95)
    plt.show()
    # Intervalul obtinut : [-1.613, 2.158]
    # iar media este ~ 0.21
    # evident, intervalul obtinut il contine pe 0
    # Daca la subpunctul anterior beta_2 lua doar pentru unele cazuri valoarea 0,
    # aici, beta_3 ia valoarea 0 pentru majoritatea cazurilor,
    # deci putem spune ca variabila premium NU ESTE un predictor util


if __name__ == "__main__":
    main()
