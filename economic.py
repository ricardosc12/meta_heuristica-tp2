from data2 import maquinas, potencias
from math import sin, pi, pow

def P1(p1,data):
    pmin,pmax,a,b,c,e,f = data
    return a*pow(p1,2) + b*p1 + c + abs(e*sin(f*(pmin-p1)))

