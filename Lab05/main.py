import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

data = pd.read_csv('trafic.csv')

valori_trafic = data["nr. masini"].values
minutes = data["minut"].values

model = pm.Model()

with model:
    possible_lambda = pm.Gamma('possible_lambda', alpha=1, beta=1)
    trafic = pm.Poisson('trafic', mu=possible_lambda, observed=valori_trafic)

with model:
    trace = pm.sample(1000, cores=1)

df = trace.to_dataframe(trace)

change_points = pm.discrete_change_points(possible_lambda, trace['possible_lambda'], threshold=0.1)

intervals = [(0, change_points[0])]

# Aflam intervalele in care lambda se schimba celmai "semnificativ"
for i in range(1, len(change_points)):
    intervals.append((change_points[i-1], change_points[i]))

intervals.append((change_points[-1], len(valori_trafic)))

lambda_means = []

for start_minute, end_minute in intervals:
    interval_data = valori_trafic[start_minute:end_minute]
    lambda_mean = np.mean(trace['possible_lambda'][start_minute:end_minute])
    lambda_means.append(lambda_mean)

for i, (start_minute, end_minute) in enumerate(intervals):
    print(f"Intervalul {i + 1}:")
    print(f"Valoarea lambda : {lambda_means[i]}")
    print(f"Min start {start_minute}")
    print(f"Min final {end_minute}")
    print()
