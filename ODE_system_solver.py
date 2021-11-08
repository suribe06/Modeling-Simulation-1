from scipy import optimize
import numpy as np
import math

def euler_explicit_method(f, t_vector, u0, h):
    n = len(t_vector)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    x3 = np.zeros(n)
    x1[0] = u0[0]
    x2[0] = u0[1]
    x3[0] = u0[2]
    for i in range(n-1):
        u = [x1[i], x2[i], x3[i]]
        x1[i + 1] = x1[i] + h * (f(u, t_vector[i])[0])
        x2[i + 1] = x2[i] + h * (f(u, t_vector[i])[1])
        x3[i + 1] = x3[i] + h * (f(u, t_vector[i])[2])
    ans = [x1, x2, x3]
    return ans

def euler_implicit_method(f, t_vector, u0, h):
    n = len(t_vector)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    x3 = np.zeros(n)
    x1[0] = u0[0]
    x2[0] = u0[1]
    x3[0] = u0[2]
    for i in range(n-1):
        u = [x1[i], x2[i], x3[i]]
        def equations(vars):
            #nonlinear equation system
            y1, y2, y3 = vars
            eq1 = y1 - x1[i] - (h * (f(vars, t_vector[i])[0]))
            eq2 = y2 - x2[i] - (h * (f(vars, t_vector[i])[1]))
            eq3 = y3 - x3[i] - (h * (f(vars, t_vector[i])[2]))
            return [eq1, eq2, eq3]
        y1, y2, y3 =  optimize.fsolve(equations, (x1[i], x2[i], x3[i]))
        x1[i + 1] = y1
        x2[i + 1] = y2
        x3[i + 1] = y3
    ans = [x1, x2, x3]
    return ans

def trapezoid_method(f, t_vector, u0, h):
    n = len(t_vector)
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    x3 = np.zeros(n)
    x1[0] = u0[0]
    x2[0] = u0[1]
    x3[0] = u0[2]
    for i in range(n-1):
        u = [x1[i], x2[i], x3[i]]
        def equations(vars):
            #nonlinear equation system
            y1, y2, y3 = vars
            eq1 = y1 - x1[i] - ((h/2)*(f(u, t_vector[i])[0] + f(vars, t_vector[i])[0]))
            eq2 = y2 - x2[i] - ((h/2)*(f(u, t_vector[i])[1] + f(vars, t_vector[i])[1]))
            eq3 = y3 - x3[i] - ((h/2)*(f(u, t_vector[i])[2] + f(vars, t_vector[i])[2]))
            return [eq1, eq2, eq3]
        y1, y2, y3 =  optimize.fsolve(equations, (x1[i], x2[i], x3[i]))
        x1[i + 1] = y1
        x2[i + 1] = y2
        x3[i + 1] = y3
    ans = [x1, x2, x3]
    return ans
