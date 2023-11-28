from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# A -> are loc un cutremur
# B -> un incendiu se declanseaza
# c -> alarma de incendiu se activeaza

# P(A) = 0.05% = 0.0005
# P(B) = 1% = 0.01 ; P(B | A) = 3% = 0.03 (muchie intre A si B)
# P(C) = 0.01% = 0.0001 ; P(C|A) = 2% = 0.02 (muchie intre A si C) ; P( C | B) = 95% = 0.95 (muchie intra B si C) ;
# P(C | A, B) = 98% = 0.98 muchie intre B si C si A si C)

deposit_model = BayesianNetwork([('A', 'B'), ('A', 'C'), ('B', 'C')])

pos = nx.circular_layout(deposit_model)
nx.draw(deposit_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

cpd_a = TabularCPD(variable='A', variable_card=2,
                   values=[[1 - 0.0005], [0.0005]])  # Are loc cutremur(0) = 0.0005, A(1) = nu are loc

# se declanseaza incendiu + se declanseaza incendiu | a avut loc cutremur
cpd_b = TabularCPD(variable='B', variable_card=2,
                   values=[[1 - 0.01, 1 - 0.03],
                           # nu se declanseaza incendiu, si nu are loc cutremur, nu se declanseaza incendiu | are loc cutremur
                           [0.01, 0.03]],
                   # se declanseaza incendiu si nu are loc cutremur, se declanseaza incendiu | are loc cutremur
                   evidence=['A'],
                   evidence_card=[2])

cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[1 - 0.0001, 1 - 0.95, 1 - 0.02, 1 - 0.98],
                           [0.0001, 0.95, 0.02, 0.98]],  # A=0,B=0; eA=0,B=1; A=1,B=0; A=1,b=1
                   evidence=['A', 'B'],
                   evidence_card=[2, 2])

deposit_model.add_cpds(cpd_a, cpd_b, cpd_c)

# assert deposit_model.check_model()

infer = VariableElimination(deposit_model)
result_1 = infer.query(variables=['A'], evidence={'C': 1})
result_2 = infer.query(variables=['B'], evidence={'C': 0})

print(result_1)

print(result_2)
