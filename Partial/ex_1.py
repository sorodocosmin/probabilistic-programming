from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
import random

def ex_1_1():
    nr_simulari = 20_000
    probabilitate_stema_jucator_0 = 1/3
    probabilitate_stema_jucator_1 = 1/2

    jucator0_castiga = 0
    jucator1_castiga = 0


    #simulam jocul de 20_000 de ori
    for _ in range(nr_simulari):
        # aruncare prima moneda:
        prima_moneda = random.random()

        player_0_started = False
        if prima_moneda < 0.5: # incepe jucatorul 0
            player_0_started = True
            # jucatorul 0 obtine stema cu probabilitate 1/3
            stema_jucator_0 = random.random()
            if stema_jucator_0 < probabilitate_stema_jucator_0:
                n = 1
            else:
                n = 0
        else: # incepe jucatorul 1
            # jucatorul 1 obtine stema cu probabilitate 1/2
            stema_jucator_1 = random.random()
            if stema_jucator_1 < probabilitate_stema_jucator_1:
                n = 1
            else:
                n = 0

        # a 2-a runda
        m = 0
        for i in range(n+1):
            if player_0_started:
                stema_jucator_1 = random.random()
                if stema_jucator_1 < probabilitate_stema_jucator_1:
                    m += 1
                else:
                    m += 0
            else: #jucatorul 1 a inceput
                stema_jucator_0 = random.random()
                if stema_jucator_0 < probabilitate_stema_jucator_0:
                    m += 1
                else:
                    m += 0

        # vedem cine a castigat
        if n >= m:  # daca jucatorul care a inceput a obtinut mai multe steme decat celalalt
            if player_0_started:
                jucator0_castiga += 1
            else:
                jucator1_castiga += 1
        else:
            if player_0_started:
                jucator1_castiga += 1
            else:
                jucator0_castiga += 1

    print(f"In simularea experimentului de {nr_simulari} ori, jucatorul 0 a castigat de {jucator0_castiga} ori, jucatorul 1 a castigat de {jucator1_castiga} ori")
    if jucator0_castiga > jucator1_castiga:
        print("Deci, Jucatorul 0 are mai multe sanse sa castige")
    elif jucator1_castiga > jucator0_castiga:
        print("Deci, Jucatorul 1 are mai multe sanse sa castige")
    else:
        print("Deci, Jucatorii au sanse egale sa castige")

def ex_1_2():

    model = BayesianNetwork([('decide_cine_incepe', 'prima_runda'),
                             ('decide_cine_incepe', 'a_doua_runda'),
                             ('prima_runda', 'a_doua_runda')  # a2-a runda este conditionata de prima
                             ])

    cpdr_decide_cine_incepe = TabularCPD('decide_cine_incepe', 2, [[0.5], [0.5]])

    cpdr_prima_runda = TabularCPD('prima_runda', 2, [[1/3, 0.5], [2/3, 0.5]], evidence=['decide_cine_incepe'], evidence_card=[2])
                                                                            # daca incepe primul player -> 1/3
    cpdr_a_doua_runda = TabularCPD('a_doua_runda', 2, [[1/3, 2/3, 0.5, 0.5], [2/3, 1/3, 0.5, 0.5]], evidence=['prima_runda', 'decide_cine_incepe'], evidence_card=[2, 2])

    model.add_cpds(cpdr_decide_cine_incepe, cpdr_prima_runda, cpdr_a_doua_runda)

    # verificam daca modelul este valid
    assert model.check_model()

    # afisam modelele CPD
    pos = nx.circular_layout(model)
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
    plt.show()

    #Inferenta -- ex1.3
    inferenta = VariableElimination(model)
    prob_prima_runda = inferenta.query(variables=['prima_runda'],
                                       evidence={'a_doua_runda': 0})
    # probabilitate fata prima runda, stiind ca nr de steme in a 2-a rund este 0
    print(prob_prima_runda)
    prob_ban = prob_prima_runda.values[0]
    prob_stema = prob_prima_runda.values[1]
    if prob_ban > prob_stema:
        print("Este mai probabil ca in prima runda sa se fi obtinut ban, stiind ca in a 2-a runda nu s-a obtinut stema")
    else:
        print("Este mai probabil ca in prima runda sa se fi obtinut stema, stiind ca in a 2-a runda nu s-a obtinut stema")


ex_1_1()
ex_1_2()
