from numeric_methods_class import NumericMethods
import math

def f(t,u):
    return -10*u

def sol_analitica(t):
    return math.exp(-10*t)

#Problem Conditions
y0 = 1
t0 = 0
tf = 1
N = 1000
nm = NumericMethods(f, y0, t0, tf, N, sol_analitica)

method_selector = 1#1:euler_explicit, 2:euler_implicit, 3:trapezoid
K = 12
norm = 2
nm.numerical_refinement_study(method_selector, K, norm)
