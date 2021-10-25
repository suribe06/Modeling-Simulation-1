from numeric_methods_class import NumericMethods
import numpy as np
import math

def f(u,t):
    return -10*u

def sol_analitica(t):
    return math.exp(-10*t)

#Problem Conditions
y0 = 1
t0 = 0
tf = 1
N = 1000
nm = NumericMethods(f, y0, t0, tf, N, sol_analitica)

#Prueba Metodos
p1 = nm.euler_explicit(True)
p2 = nm.euler_implicit(True)
p3 = nm.trapezoid(True)

method_selector = 1 #0:euler_explicit, 1:euler_implicit, 2:trapezoid, 3:RK Implicit, 4:RK Explicit
norm = 2
#Prueba del analisis de convergencia
c, p = nm.convergence_analysis(method_selector, norm)
print(c)
print(p)

#Prueba Refinamiento Numerico
K = 12
hs, E_sequence = nm.numerical_refinement_study(method_selector, K, norm, True)

#Matrices para RK
A = np.array([[0,0,0,0], [1/2,0,0,0], [0,1/2,0,0], [0,0,1,0]])
B = [1/6, 1/3, 1/3, 1/6]
C = [0, 1/2, 1/2, 1]
A2 = [[1/4, (3 - 2*math.sqrt(3))/12], [(3 + 2*math.sqrt(3))/12, 1/4]]
B2 = [1/2, 1/2]
C2 = [(3 - math.sqrt(3))/6, (3 + math.sqrt(3))/6]

#Prueba Refinamiento Numerico RK Implicito
c, p = nm.convergence_analysis(3, norm, A2, B2, C2)
print(c)
print(p)

#Prueba Refinamiento Numerico RK Explicito
hs, E_sequence = nm.numerical_refinement_study(4, K, norm, True, A, B, C)
