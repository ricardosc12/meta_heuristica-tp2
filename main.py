from math import sin, pi, pow
import random
import numpy as np

F = 0.8
Cr = 0.9

max = [10,10]
min = [0 ,0]

# def limits(solution):

def g1(xVector):
    x1 = xVector[0]
    x2 = xVector[1] 
    g1x = pow(x1,2) - x2 + 1 # <= 0
    return True if g1x<=0 else False

def g2(xVector):
    x1 = xVector[0]
    x2 = xVector[1] 
    g2x = 1 - x1 + pow((x2 - 4),2) # <= 0
    return True if g2x<=0 else False

def restricoes(xVector):
    g1x = g1(xVector)
    g2x = g2(xVector)
    
    # fact = pow(g1x,2) + pow(g2x,2)

    return True if g1x and g2x else False

def f(xVector):
    x1 = xVector[0]
    x2 = xVector[1] 

    numerador = pow(sin(2*pi*x1),3)*sin(2*pi*x2)
    denominador = pow(x1,3)*(x1+x2)

    return -(numerador/denominador)

def _fitness(vetor):
    return np.sum(list(map(lambda x: f(x),vetor)))

def randFloat():
    return random.uniform(0,1)

def _random(population):
    return population[random.randint(0, len(population)-1)]

def best(population):
    best = population[0]
    for solution in population:
        if(f(solution)<=f(best)):
            best = solution
    return best

def generation(NP):
    population = []
    intervalX = (max[0] - min[0])/NP
    intervalY = (max[1] - min[1])/NP

    for i in range(NP+1):
        # x = 0.1 if i*intervalX == 0 else i*intervalX
        y = i*intervalY
        _min = pow((y-4),2)+1
        _max = pow(abs((y-1)),1/2)

        x = random.uniform(_min, _max)

        population.append([x if x<max[0] else max[0],y])

    return population
    
def cruzamento(target,doador):
    result = [0 for x in range(len(target))]
    for i in range(len(target)):
        if(randFloat()<=Cr):
            result[i] = doador[i]
        else: result[i] = target[i]
    return result

def g3(xVector):
    return True if pow(xVector[0],3)*(xVector[0]+xVector[1])>0 else False

def gg(xVector):
    return g1(xVector) and g2(xVector) and g3(xVector)

def ED (F, Cr, NP):
    vetor = generation(NP) # População inicial  [[0,0], [0.5,0.5], [2,2]]
    
    f_vetor = _fitness(vetor)

    # f_vetor = f(vetor)   # Fitness da população  [10, 20, 40]
    geration = 0

    nova_geracao = [0 for i in range(NP)]

    while True:
        for i in range(NP):

            vetor_doador = []

            while not len(vetor_doador):
                R3 = np.array(best(vetor))
                R1 = np.array(_random(vetor))
                R2 = np.array(_random(vetor))

                # print('R3',R3)
                # print('R1',R1)
                # print('R2',R2)
                aux = R3 + F*(R1 - R2)

                vetor_doador = aux if gg(aux) else [] # Mutação diferencial # R3 _random ou best

            # print('vetor_doador',vetor_doador)
            vetor_target = []
            while not len(vetor_target):
                aux = _random(vetor)
                vetor_target = aux if gg(aux) else []


            # print('vetor_target',vetor_target)

            

            vetor_trial = cruzamento(vetor_target, vetor_doador)  # Cruzamento Binomial ou Exponencial

            vetor_trial = []
            while not len(vetor_trial):
                aux = _random(vetor)
                vetor_trial = aux if gg(aux) else []

            # print('vetor_trial',vetor_trial)

            if(f(vetor_trial) <= f(vetor_target)): 
                nova_geracao[i] = vetor_trial
            else:
                nova_geracao[i] = vetor_target

        geration += 1

        if(_fitness(vetor) <= _fitness(nova_geracao)):

            # print('vetor',best(vetor))
            # print('result',f(best(vetor)))
            return f(best(vetor)),best(vetor)

        vetor = nova_geracao

# ED(F,Cr,20)
_BEST,_VARS = ED(F,Cr,20)
for i in range(30):
    aux1,aux2 = ED(F,Cr,20)
    if(aux1<=_BEST):
        _BEST =aux1
        _VARS =aux2

print(_BEST,_VARS)

print('confirm',f(_VARS),_VARS)
print('base',f([1.22,4.70]),[1.22,4.70])

# Minimize

# Subject to:




# Sendo 0 ≤ 𝑥1 ≤ 10 e 0 ≤ 𝑥2 ≤ 10
# print(f([1,2]))