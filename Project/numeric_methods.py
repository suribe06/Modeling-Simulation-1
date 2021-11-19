from scipy import optimize
import numpy as np
import math

def euler_explicit_method(f, t_vector, u0, h):
    n = len(t_vector)
    G = np.zeros(n)
    S = np.zeros(n)
    T = np.zeros(n)
    G[0] = u0[0]
    S[0] = u0[1]
    T[0] = u0[2]
    for i in range(n-1):
        u = [G[i], S[i], T[i]]
        G[i + 1] = G[i] + h * (f(u, t_vector[i])[0])
        S[i + 1] = S[i] + h * (f(u, t_vector[i])[1])
        T[i + 1] = T[i] + h * (f(u, t_vector[i])[2])
    ans = [G, S, T]
    return ans

def euler_implicit_method(f, t_vector, u0, h):
    n = len(t_vector)
    G = np.zeros(n)
    S = np.zeros(n)
    T = np.zeros(n)
    G[0] = u0[0]
    S[0] = u0[1]
    T[0] = u0[2]
    for i in range(n-1):
        u = [G[i], S[i], T[i]]
        def equations(vars):
            #nonlinear equation system
            y1, y2, y3 = vars
            eq1 = y1 - G[i] - (h * (f(vars, t_vector[i])[0]))
            eq2 = y2 - S[i] - (h * (f(vars, t_vector[i])[1]))
            eq3 = y3 - T[i] - (h * (f(vars, t_vector[i])[2]))
            return [eq1, eq2, eq3]
        y1, y2, y3 =  optimize.fsolve(equations, (G[i], S[i], T[i]))
        G[i + 1] = y1
        S[i + 1] = y2
        T[i + 1] = y3
    ans = [G, S, T]
    return ans

def trapezoid_method(f, t_vector, u0, h):
    n = len(t_vector)
    G = np.zeros(n)
    S = np.zeros(n)
    T = np.zeros(n)
    G[0] = u0[0]
    S[0] = u0[1]
    T[0] = u0[2]
    for i in range(n-1):
        u = [G[i], S[i], T[i]]
        def equations(vars):
            #nonlinear equation system
            y1, y2, y3 = vars
            eq1 = y1 - G[i] - ((h/2)*(f(u, t_vector[i])[0] + f(vars, t_vector[i])[0]))
            eq2 = y2 - S[i] - ((h/2)*(f(u, t_vector[i])[1] + f(vars, t_vector[i])[1]))
            eq3 = y3 - T[i] - ((h/2)*(f(u, t_vector[i])[2] + f(vars, t_vector[i])[2]))
            return [eq1, eq2, eq3]
        y1, y2, y3 =  optimize.fsolve(equations, (G[i], S[i], T[i]))
        G[i + 1] = y1
        S[i + 1] = y2
        T[i + 1] = y3
    ans = [G, S, T]
    return ans
