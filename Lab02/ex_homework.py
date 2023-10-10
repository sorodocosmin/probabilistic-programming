import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import random

def ex_1 ():
    """
        Simulam scenariul unui service, iar noi il putem monitoriza ca o distributie exponentiala
        unul este mai rapid decat celalalt;
        generam 10_000 valori noi
        vrem sa stim, in medie cat timp asteptam atunci cand mergem la service
        avem 2 distributii si trebuie combinate pentru a obtine distributia pentru client. timpul de asteptare pt client

    """
    np.random.seed(1)

    lambda_1 = 4.0
    lambda_2 = 6.0
    val_mecanic_1 = stats.expon(0,1/lambda_1).rvs(10_000)
    val_mecanic_2 = stats.expon(0,1/lambda_2).rvs(10_000)

    #alegem random din cele 2 distributii
    # cu 40% sansa sa alegem din prima distributie
    i = 0
    X = []
    while i < 10_000:
        nr_random = random.randint(1,100)
        i += 1
        if nr_random <= 40:
            X.append(random.choice(val_mecanic_1))
        else:
            X.append(random.choice(val_mecanic_2))

    az.plot_posterior({'mecanic_1':val_mecanic_1,
                       'mecanic_2':val_mecanic_2,
                       'client':X})
    plt.show()

def ex_2 ():
    """
    4 servere, iar fiecare are un latency diferit
    avem nev de acea distributie gama, pt a afla cat trebuie sa asteptam pentru a ni se raspunde la acel request
    asemanator cu primul exercitiu; trebuie sa gasim un mod de a le combina
    """
    latenta = stats.expon(0,1/4).rvs(10_000)
    
    server_1 = stats.gamma(4,0,1/3).rvs(10_000)
    server_2 = stats.gamma(4,0,1/2).rvs(10_000)
    server_3 = stats.gamma(5,0,1/2).rvs(10_000)
    server_4 = stats.gamma(5,0,1/3).rvs(10_000)

    X = []
    i = 0
    while( i < 10_000):
        probability = random.randint(1,100)
        i += 1
        if probability <= 25:
            X.append(random.choice(server_1) + random.choice(latenta))
        elif probability <= 50:
            X.append(random.choice(server_2) + random.choice(latenta) )
        elif probability <= 80:
            X.append(random.choice(server_3)+ random.choice(latenta))
        else:
            X.append(random.choice(server_4)+ random.choice(latenta))

    nr_bigger_than_3 = len([1 for i in X if i > 3])
    print(f"The probability of the response to be bigger than 3ms is : {nr_bigger_than_3/10_000}")
    az.plot_posterior({'servire_client' : X})
    plt.show()

def ex_3 ():
    """
    trebuie generate niste valori ( pt a da cu banul)

    """
    gen_valori = [0,0,0,0] # gen_valori[0] - reprezinta ss, [1] - sb, [2] -bs ; [3] - bb
    ss = []
    sb = []
    bs = []
    bb = []
    i = 0
    while i < 100:
        #aruncam cu banii de 10 ori
        for _ in range(10):
            nr_random_ban_1 = random.randint(1,2)
            nr_random_ban_2 = random.randint(1,100)
            if nr_random_ban_2 <= 30 and nr_random_ban_1 ==0 : #obtinem ss
                ss.append(1)
                sb.append(0)
                bs.append(0)
                bb.append(0)
            elif nr_random_ban_2 < 30 and nr_random_ban_1 == 1 : #obtinem bs
                bs.append(1)
                ss.append(0)
                sb.append(0)
                bb.append(0)
            elif nr_random_ban_2 > 30 and nr_random_ban_1 == 0 : #obtinem sb
                sb.append(1)
                ss.append(0)
                bs.append(0)
                bb.append(0)
            else: #obtinem bb
                bb.append(1)
                sb.append(0)
                bs.append(0)
                ss.append(0)
        i += 1

    az.plot_posterior({'ss':ss,
                          'sb':sb,
                          'bs':bs,
                          'bb':bb})
    plt.show()


# ex_1()
# ex_2()
# ex_3()
ex_3()

