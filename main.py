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

nm.euler_implicit(True)

method_selector = 1 #0:euler_explicit, 1:euler_implicit, 2:trapezoid, 3:RK Implicit, 4:RK Explicit
norm = 2
#Numerical refinement study
K = 12
nm.numerical_refinement_study(method_selector, K, norm)
#Convergence analysis
c, p = nm.convergence_analysis(method_selector, norm)
print(c)
print(p)
