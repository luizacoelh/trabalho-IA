import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# 1. Definir as variáveis de entrada e saída
# As variáveis são definidas com seus respectivos universos (intervalos de valores)
temperatura = ctrl.Antecedent(np.arange(0, 41, 1), 'temperatura')
pessoas = ctrl.Antecedent(np.arange(0, 21, 1), 'pessoas')
conforto = ctrl.Consequent(np.arange(0, 11, 1), 'conforto')

# 2. Criar as funções de pertinência com trimf (triangular)
# A documentação menciona o uso de `trimf` para as funções triangulares.
temperatura['fria'] = fuzz.trimf(temperatura.universe, [0, 0, 20])
temperatura['amena'] = fuzz.trimf(temperatura.universe, [15, 25, 35])
temperatura['quente'] = fuzz.trimf(temperatura.universe, [30, 40, 40])

pessoas['poucas'] = fuzz.trimf(pessoas.universe, [0, 0, 10])
pessoas['media'] = fuzz.trimf(pessoas.universe, [5, 10, 15])
pessoas['lotado'] = fuzz.trimf(pessoas.universe, [10, 20, 20])

conforto['baixo'] = fuzz.trimf(conforto.universe, [0, 0, 5])
conforto['medio'] = fuzz.trimf(conforto.universe, [3, 5, 7])
conforto['alto'] = fuzz.trimf(conforto.universe, [5, 10, 10])

# O documento menciona a plotagem com matplotlib [cite: 4]
# Para a apresentação, você pode mostrar os gráficos
# temperatura.view()
# pessoas.view()
# conforto.view()

# 3. Definir as regras de inferência
# As regras são baseadas no documento [cite: 17, 18, 19, 20, 21, 22, 23]
regra1 = ctrl.Rule(temperatura['quente'] & pessoas['lotado'], conforto['baixo'])
regra2 = ctrl.Rule(temperatura['amena'] & pessoas['media'], conforto['alto'])
regra3 = ctrl.Rule(temperatura['fria'] & pessoas['poucas'], conforto['medio'])

# 4. Criar e simular o sistema de controle
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])
simulador = ctrl.ControlSystemSimulation(sistema_controle)

# 5. Fornecer as entradas
# Usamos o cenário do exercício 1 
simulador.input['temperatura'] = 30
simulador.input['pessoas'] = 10

# 6. Calcular o resultado
simulador.compute()

# 7. Imprimir e visualizar o resultado
print("Temperatura de entrada: 30")
print("Pessoas de entrada: 10")
print(f"Grau de pertinência para 'conforto baixo': {fuzz.interp_membership(conforto.universe, conforto['baixo'].mf, simulador.output['conforto'])}")
print(f"Resultado do Conforto: {simulador.output['conforto']}")

conforto.view(sim=simulador)
plt.show()