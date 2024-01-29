import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

def main():
    # Ex - 1 - a
    df = pd.read_csv("Examen\\Titanic.csv")
    # stergem coloanele care nu ne intereseaza
    df = df.drop(["PassengerId", "Name", "Sex", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

    # stergem randurile care au valori lipsa
    df = df.dropna()

    output_survived = df["Survived"].values

    #pentru a ne face o idee asupra mediilor si dev. standard:

    # print(len(output_survived[output_survived==1]),len(output_survived[output_survived==0]))
    # # de asemenea, se observa ca datele nu sunt echilibrate 127 - 205
    # Index = np.random.choice(np.flatnonzero(output_survived==0), size=len(output_survived[output_survived==0])-len(output_survived[output_survived==1]), replace=False) #pentru a balansa datele, alegem la intamplare indici pentru a fi stersi
    # df = df.drop(labels=Index)

    col_age = df["Age"].values
    col_class = df["Pclass"].values
    age_mean = col_age.mean()
    age_std = col_age.std()

    # standardizam datele pt age
    col_age = (col_age - age_mean)/age_std
    # pt class, atribut discret, nu facem standardizarea

    X = np.column_stack((col_age, col_class))
    X_mean = X.mean(axis=0, keepdims=True)

    print(f"Media var independente : {X_mean}")   
    print(f"Media output {output_survived.mean()}")
    print(f"Dev standard {X.std(axis=0, keepdims=True)}")
    print(f"Output dev standard : {output_survived.std()}")

    # ex b)
    with pm.Model() as model_mlr:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1, shape=2)
        X_shared = pm.MutableData('x_shared',X)
        miu = pm.Deterministic('miu',alpha + pm.math.dot(X_shared, beta))
        theta = pm.Deterministic('theta', pm.math.sigmoid(miu))

        bd = pm.Deterministic('bd', -alpha/beta[1] + beta[0]/beta[1] * X_shared[:,0].mean())

        y_pred = pm.Bernoulli('y_pred', p=theta, observed=output_survived)

        idata = pm.sample(1250, return_inferencedata=True)

    # ex 1 - c
    # obtinem intervalul de incredere pentru beta
    # pentru a determina care variabila are o influenta mai mare asupra output-ului
    az.plot_forest(idata, hdi_prob=0.95, var_names=['beta'])
    plt.show()
    print(az.summary(idata, hdi_prob=0.95, var_names=['beta']))
    # se poate observa ca beta[0] (age) nu are o inluenta atat de mare asupra putputului, comparativ cu
    # beta[1] (class), care are o influenta mai mare asupra output-ului
    # de asemenea, se poate observa ca beta[0] contine 0 in intervalul sau;
    # deci, variabila care influenteaza cel mai mult output-ul este class

    # d)
    # persoana de 30 de ani(care va fi standardizata), de la clasa a2-a
    obs_std2 = [(30-age_mean)/age_mean, 2]
    pm.set_data({"x_shared":[obs_std2]}, model=model_mlr)
    ppc = pm.sample_posterior_predictive(idata, model=model_mlr,var_names=["theta"])
    y_ppc = ppc.posterior_predictive['theta'].stack(sample=("chain", "draw")).values
    # construim intervalul de incredere de 90%, daca va supravietui sau nu
    az.plot_posterior(y_ppc,hdi_prob=0.9)
    plt.show()
    # Din graffic se poate observa ca o astfel de persoana are o probabilitate de 39% sa supravietuiasca


    

if __name__=="__main__":
    main()