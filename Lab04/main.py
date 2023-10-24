import numpy as np
from scipy.stats import poisson, norm, expon


lambda_client = 20
medie_timp_plasare_si_plata = 2
deviatie = 0.5

nr_simulari = 100_000

# client_poisson = poisson.rvs(lambda_client, size=nr_simulari)
# timp_plasare_si_plata_norm = norm.rvs(medie_timp_plasare_si_plata, deviatie, size=nr_simulari)
#
# alpha = 5
# timp_pregatire_comanda_expon = expon.rvs(0, alpha, size=nr_simulari)

# ar trebui sa generam timp_pregatire_expon si sa il crestem pe alfa atat timp cat
start_alpha = 1
step_alpha = 0.1

timp_pregatire_comanda_expon = expon.rvs(0, start_alpha, size=nr_simulari)
timp_plasare_si_plata_norm = norm.rvs(medie_timp_plasare_si_plata, deviatie, size=nr_simulari)

timp_servire = timp_pregatire_comanda_expon + timp_plasare_si_plata_norm
probabilitate_client_servit_sub_15_m = np.mean(timp_servire < 15)

while probabilitate_client_servit_sub_15_m >= 0.95:
    start_alpha += step_alpha
    timp_pregatire_comanda_expon = expon.rvs(0, start_alpha, size=nr_simulari)
    timp_plasare_si_plata_norm = norm.rvs(medie_timp_plasare_si_plata, deviatie, size=nr_simulari)
    timp_servire = timp_pregatire_comanda_expon + timp_plasare_si_plata_norm

    probabilitate_client_servit_sub_15_m = np.mean(timp_servire < 15)

start_alpha -= step_alpha
print(f"Alpha maxim pentru care 95% din clienti sunt serviti intr-un timp mai scurt de 15 minute este = {start_alpha}")

timp_pregatire_comanda_expon = expon.rvs(0, start_alpha, size=nr_simulari)
timp_plasare_si_plata_norm = norm.rvs(medie_timp_plasare_si_plata, deviatie, size=nr_simulari)
timp_servire = timp_pregatire_comanda_expon + timp_plasare_si_plata_norm

print(f"Media de asteptare pentru a fi servit un client este : {np.mean(timp_servire)} minute")

