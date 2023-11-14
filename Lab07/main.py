import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import arviz as az

df = pd.read_csv('auto-mpg.csv')


#print(df.describe())

# trebuie sa eliminam acele valori pentru care apare '?'
# curatam datele
df = df[df['horsepower'] != '?']
df = df[df['mpg'] != '?']

# convertim coloanele horsepower si mpg la tip numeric
df['horsepower'] = pd.to_numeric(df['horsepower'])
df['mpg'] = pd.to_numeric(df['mpg'])


#print(df.describe())
plt.figure(figsize=(10, 6))
plt.scatter(df['horsepower'], df['mpg'])
plt.title('Relația dintre  și mpg')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Consum de carburant (mpg)')
plt.grid(True)
plt.show()

# definirea modelului folosindu-ne de pymc
X = np.array(df['horsepower'])
Y = df['mpg']

model = pm.Model()

with model:
    alpha = pm.Normal(name='alpha', mu=0)
    beta = pm.Normal(name='beta', mu=0)
    miu = pm.Deterministic('miu', alpha + beta * X)
    # Definirea distributiei pentru observatii
    mpg_predicted = pm.Normal(name='mpg_predicted', mu=miu, observed=Y)

with model:
    # antrenarea
    trace = pm.sample(2000, tune=1000)

# Dreapta de regresie
alpha_mean = trace['alpha'].mean()
beta_mean = trace['beta'].mean()

print(f'Dreapta de regresie: mpg = {alpha_mean} + {beta_mean} * CP')


az.plot_hdi(X, trace['miu'], hdi_prob=0.95, color='red')
plt.scatter(df['horsepower'], df['mpg'])
plt.title('Relația dintre caii putere și consumul de carburant cu HDI')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Consum de carburant (mpg)')
plt.grid(True)
plt.show()


"""
    Se poate observa pe baza graficului, cu cat numarul de cai putere este mai mare,
    cu atat consumul scade.
"""
