from scipy.stats import geom
import matplotlib.pyplot as plt
import numpy as np

def estimate():
    """
    returneaza probabilitatea ca X > Y^2
    """
    N = 10_000
    # theta pentru X si Y se dau in cerinta
    theta_X = 0.3
    theta_Y = 0.5
    # simulam variabilele aleatoare repartizate Geometric, cu parametrii theta_X si theta_Y
    x = geom.rvs(theta_X, size=N)
    y = geom.rvs(theta_Y, size=N)


    # print(len(x),len(y))
    condition = x > y**2
    # numaram pentru valorile simulate pentru variabilele aleatoare
    # de cate ori este "respectata" conditia
    count_condition = sum(condition)
    # print(len(x[condition]))
    # print(len(x[~condition]))
    probability_estimate = count_condition / N
    # calculam probabilitatea ca fiind nr de valori care respecta conditia / nr total de valori simulate

    # codul de mai jos este comentat intrucat aiseaza pe un plot valorile generate
    # cu albastru pentru valorile care respecta conditia si cu rosu pentru cele care nu o respecta
    # a fost comentat intrucat pt k = 30, s-ar fi generat prea multe grafice
    # graficul incarcat a fost obtinut dand pentru k = 1

    # plt.figure(figsize=(6, 6))
    # plt.plot(x[condition], y[condition], 'b.')
    # plt.plot(x[~condition], y[~condition], 'r.')
    # plt.plot(0, 0, label=f'P(X > Y^2) = {probability_estimate:.4f}')
    # plt.title('Simulare variabile aleatoare geometrice')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.show()

    return probability_estimate

def main():
    # ex - 2 -b
    k = 30
    vals = []
    # trebuie sa simulam acest experiment de k ori
    # introducem intr-un vector valorile obtinute pt fiecare estimare
    # ,iar la final afisam media si deviatia standard pentru aceste probabilitati
    for _ in range(k):
        vals.append(estimate())
    
    print(f"Probabilitatea ca X > Y^2 este {np.mean(vals)} pentru {k} realizari")
    print(f"Deviatia standard este {np.std(vals)} pentru {k} realizari")

main()