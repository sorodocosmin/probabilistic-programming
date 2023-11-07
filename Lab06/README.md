# Efectul lui Y si 	θ
 
- **Y** reprezinta numarul de clienti care cumpara un anumit produs
- **θ** reprezinta probabilitate cu care un client compara acel produs

Analizand graficele rezultate in urma rularii programului, se pot observa
anumite schimbari a numarului de clienti care viziteaza magazinul in functie de
aceste doua variabile.

- evident, daca numarul de clienti care cumpara un anumit produs creste (Y creste),
atunci si numarul de clienti care viziteaza magazinul creste (n din distributia Poisson ).
Este normal ca atunci cand numarul de clienti care **cumpara** este mai mare, ca si numarul de
clienti care **viziteaza** sa fie mai mare. (acest lucru se poate observa si in graficele din partea stanga)
- daca probabilitate cu care un client cumpara un anumit produs este mai mare (θ creste),
atunci numarul de clienti care viziteaza acel magazin este mai mic (n din distributia Poisson).
_De ce se intampla acest lucru ?_ Pentru a intelege mai bine, prezentam in continuare un exemplu
  \
(\
Presupunem ca _y_ este numarul de clienti care au cumparat un produs, iar fiecare astfel de client, cumpara un produs
cu probabilitatea _p_ (Daca _p_ ar fi = 1/2, atunci, in medie, 1 din 2 clienti care au intrat in magazin, au si cumparat un produs, 
Daca _p_ ar fi 1/10, atunci in medie 1 din 10 oameni care intra in magazin, cumpara un produs. Deci, un numar mai mare de clienti
care viziteaza magazinul)
&rarr; y = p1 * n1; y = p2 * n2;\
Evident, daca p1 > p2, atunci n1 < n2.\
)

Astfel, cu cat numarul de clienti care cumpara un produs este **mai mare**, si probabilitatea cu care il cumpara este **mai mica**,
cu atat **numarul de clienti care viziteaza magazinul (n)** este _mai mare_.