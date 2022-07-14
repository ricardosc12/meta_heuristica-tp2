# from data2 import maquinas, potencias
# from caso2 import maquinas
from caso3 import maquinas, potencias
from math import sin, pi, pow
import numpy as np
import random


PD = 1800
ITERATIONS = 200
GERATIONS = 20
NP = 10
Cr = 0.9
F = 0.5

def P1(p1,data):
    pmin,pmax,a,b,c,e,f = data
    return a*pow(p1,2) + b*p1 + c + abs(e*sin(f*(pmin-p1)))

def f(pi):
    def calcPI(i):
        return P1(i[1],maquinas[i[0]])
    return np.sum(list(map(calcPI,enumerate(pi))))

def _fitness(population):
    return np.sum(list(map(lambda x:f(x),population)))

def randFloat():
    return random.uniform(0,1)

def _generation(data):
    def sumMax(pos,pd):
        return pd - np.sum(list(map(lambda x:x[0],data[pos:len(data)])))
    pd = PD
    population = []

    for i in range(len(data)):
        _max = sumMax(i+1,pd)
        _max = _max if _max<data[i][1] else data[i][1]
        _min = data[i][0]
        pi = random.uniform(_min,_max)
        pd-=pi
        population.append(pi)
    
    pd_resto = PD - np.sum(population)
    while pd_resto!=0:
        _pd = pd_resto/len(data)
        for i in range(len(population)):
            _max = data[i][1] - population[i]
            _max = _max if _pd>_max else _pd
            _get_pd = random.uniform(0,_max)
            population[i]+=_get_pd

        pd_resto = PD - np.sum(population)

    return population


def generation(NP):
    population = []
    for i in range(NP):
        population.append(_generation(maquinas))
    return population

def cruzamento(target,doador):
    result = [0 for x in range(len(target))]
    for i in range(len(target)):
        if(randFloat()<=Cr):
            result[i] = doador[i]
        else: result[i] = target[i]
    return result

def g1(data):
    return True if np.sum(data)==PD else False

def g2(data):
    for i in range(len(data)):
        if(data[i]>maquinas[i][1] or data[i]<maquinas[i][0]):
            return False
    return True

def gg(data):
    return g1(data) and g2(data)

def restrictions(vetor):
    def g1(vetor):
        max_min = 0
        for i in range(len(vetor)):
            restriction = maquinas[i][0]-vetor[i]
            max_min+= 0 if restriction<0 else restriction
        return pow(max_min,2)
    def g2(vetor):
        max_min = 0
        for i in range(len(vetor)):
            restriction = vetor[i]-maquinas[i][1]
            max_min+= 0 if restriction<0 else restriction
        return pow(max_min,2)
    def g3(vetor):
        return pow(np.sum(vetor)-PD,2)
        
    return g1(vetor)+g2(vetor)+g3(vetor)

# def constrainedEpsilon(v):
#     return restrictions(v)<=0 and g3(v)

def _random(population):
    return population[random.randint(0, len(population)-1)]

def best(population):
    best = population[0]
    index = 0
    for p in range(len(population)):
        if(f(population[p])<=f(best)):
            best = population[p]
            index = p
    return best

# population = generation(20)
# for i in population:
#     print(f(i))

# best_, index = best(population)
# print('best',f(best_))


def ED (F, Cr, NP):
    vetor = generation(NP) 
    geration = 0
    nova_geracao = [0 for i in range(NP)]

    while True:
        for i in range(NP):

            R3 = np.array(best(vetor))
            R1 = np.array(_random(vetor))
            R2 = np.array(_random(vetor))
            vetor_doador = R3 + F*(R1 - R2)
            vetor_target = _random(vetor)
            vetor_trial = cruzamento(vetor_target, vetor_doador)

            next_gen = []

            if(f(vetor_trial) <= f(vetor_target)): 
                nova_geracao[i] = vetor_trial
            else:
                nova_geracao[i] = vetor_target

        geration += 1

        vetor = nova_geracao

        if(geration == GERATIONS):
            return f(best(vetor)),best(vetor)


vetor = generation(NP) 
for i in vetor:
    print(restrictions(i))
# _BEST,_VARS = ED(F,Cr,NP)
# for i in range(ITERATIONS):
#     aux1,aux2 = ED(F,Cr,NP)
#     if(aux1<=_BEST):
#         _BEST =aux1
#         _VARS =aux2

# # print(_BEST,_VARS)
# for i in _VARS:
#     print(i)

# print('confirm',f(_VARS))
# print("SOMA DE PI", np.sum(_VARS))
# print('artigo',f(potencias))