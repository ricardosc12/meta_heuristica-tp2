from math import sin, pi, pow
import random
from turtle import Vec2D
import numpy as np

F = 0.8
Cr = 0.9
NP = 30
GERATIONS = 50
ITERATIONS = 30

max = [10, 10]
min = [0 , 0]


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

    return True if g1x and g2x else False

def f(xVector):
    x1 = xVector[0]
    x2 = xVector[1] 

    numerador = pow(sin(2*pi*x1),3)*sin(2*pi*x2)
    denominador = pow(x1,3)*(x1+x2)

    return -(numerador/denominador)

def randFloat():
    return random.uniform(0,1)

def _random(population):
    return population[random.randint(0, len(population)-1)]

def bestFact(population):
    best = population[0]
    for solution in population:
        if(f(solution)<=f(best) and constrainedEpsilon(solution)):
            best = solution
    return best

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
        y = i*intervalY
        _min = pow((y-4),2)+1
        _max = pow(abs(y-1),1/2)
        x = random.uniform(_min, _max)

        population.append([x,y])

    return population
    
def cruzamento(target,doador):
    result = [0 for x in range(len(target))]
    for i in range(len(target)):
        if(randFloat()<=Cr):
            result[i] = doador[i]
        else: result[i] = target[i]
    return result

def g3(xVector):
    return True if pow(xVector[0],3)*(xVector[0]+xVector[1])!=0 else False
def limits(vetor):
    x,y = vetor
    limitX = x>=min[0] and x<=max[0]
    limitY = y>=min[1] and y<=max[1]
    return limitY and limitX

def gg(xVector):
    return g1(xVector) and g2(xVector) and g3(xVector)

def restrictions(vetor):
    def g1(vetor):
        rest = pow(vetor[0],2)-vetor[1]+1
        return 0 if rest<0 else pow(rest,2)
    def g2(vetor):
        rest = 1 - vetor[0] + pow((vetor[1] - 4),2)
        return 0 if rest<0 else pow(rest,2)
    def g3(vetor):
        x,y = vetor
        return abs(pow(x,3)*(x+y))
    return g1(vetor)+g2(vetor)
def constrainedEpsilon(v):
    return restrictions(v)<=0 and g3(v)



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

            # epsillon constrained Method
            if(constrainedEpsilon(vetor_trial) and constrainedEpsilon(vetor_target)):
                if(f(vetor_trial) <= f(vetor_target)): 
                    next_gen = vetor_trial
                else:
                    next_gen = vetor_target
            else:
                if(restrictions(vetor_trial) < restrictions(vetor_target)):
                    next_gen=vetor_trial
                else: next_gen=vetor_target
            # ------------------------
            
            nova_geracao[i] = next_gen if(constrainedEpsilon(next_gen)) else vetor[i]

        geration += 1
        vetor = nova_geracao

        if(geration == GERATIONS):
            return f(bestFact(vetor)),bestFact(vetor)


_BEST,_VARS = ED(F,Cr,NP)
for i in range(ITERATIONS):
    aux1,aux2 = ED(F,Cr,NP)
    if(aux1<=_BEST):
        _BEST =aux1
        _VARS =aux2

print(_BEST,_VARS)
print(gg(_VARS))
print(constrainedEpsilon(_VARS))
print('confirm',f(_VARS),_VARS)
print('base',f([1.22,4.70]),[1.22,4.70])
