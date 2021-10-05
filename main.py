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

method_selector = 1 #0:euler_explicit, 1:euler_implicit, 2:trapezoid, 3:RK Implicit, 4:RK Explicit
norm = 2
#Numerical refinement study
K = 12
nm.numerical_refinement_study(method_selector, K, norm)
#Convergence analysis
A = np.array([[0,0,0,0], [1/2,0,0,0], [0,1/2,0,0], [0,0,1,0]])
B = [1/6, 1/3, 1/3, 1/6]
C = [0, 1/2, 1/2, 1]
A2 = [[1/4, (3 - 2*math.sqrt(3))/12], [(3 + 2*math.sqrt(3))/12, 1/4]]
B2 = [1/2, 1/2]
C2 = [(3 - math.sqrt(3))/6, (3 + math.sqrt(3))/6]
#c, p = nm.convergence_analysis(method_selector, norm, A, B, C)
#print(c)
#print(p)
nm.runge_kutta_explicit_rs(A, B, C, True)
nm.runge_kutta_implicit_2s(A2, B2, C2, True)
