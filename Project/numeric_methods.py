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
