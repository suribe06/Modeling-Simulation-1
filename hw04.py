#Numerical refinement study

from hw02 import euler_explicit_method, euler_implicit_method, trapezoid_method, f
import matplotlib.pyplot as plt
import numpy as np
import math

def norm_p(e, p, h):
    ans = None
    if p == np.inf:
        ans = max(e)
    else:
        sum = 0
        for x in e: sum += abs(x)**p
        ans = h * (sum **(1/p))
    return ans

def main():
    N = 1000
    t0 = 0
    tf = 2*math.pi
    y0 = 1
    t_vector, h = np.linspace(t0, tf, num=N, retstep=True) #malla de muchos puntos
    ode_approx = euler_explicit_method(t_vector, y0, h)

    K = 12
    E_sequence = []
    hs = []
    for i in range(1, K+1):
        h_i = h / (2**i)
        ode_approx_i = euler_explicit_method(t_vector, y0, h_i)
        e_i = []
        for j in range(N):
            e_i.append(abs(ode_approx_i[j] - ode_approx[j]))
        p = 2
        E_i = norm_p(e_i, p, h_i)
        E_sequence.append(E_i)
        hs.append(h_i)

    print(E_sequence)
    plt.plot(hs, E_sequence, "cx", label="E Sequence")
    plt.title("Numerical Refinement Study")
    plt.ylabel("E(h)")
    plt.xlabel("h")
    plt.legend()
    plt.grid()
    plt.show()
    return

main()
