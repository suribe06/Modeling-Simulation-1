#Runge-Kutta Method for r steps

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

def runge_kutta(A, B, C, f, u0, t_vector, h):
    assert len(B) == len(C) and len(B) == len(A)
    r = len(B)
    n = len(t_vector)
    u_vector = np.zeros(n) #solution vector
    u_vector[0] = u0
    ys = [] #vector of ys
    for i in range(n-1):
        for j in range(r):
            sum = 0
            for k in range(j+1):
                if A[j][k] == 0:
                    sum += 0
                else: #A[j][k] != 0
                    if k < j: #Rung-Kutta Explicito
                        sum += A[j][k] * f(ys[j-1], t_vector[i] + (h*C[k]))
                    else: # k >= j Runge-Kutta Implicito
                        print("entre a runge-kutta implicito")
            y_i = u_vector[i] + (h * sum)
            ys.append(y_i)
        #Update next value of u_vector
        sum = 0
        for j in range(r):
            sum += B[j] * f(ys[j], t_vector[i] + (C[j] * h))
        u_vector[i + 1] = u_vector[i] + (h * sum)
    return u_vector

def main():
    #First Matrix
    B = [1/6, 1/3, 1/3, 1/6]
    C = [0, 1/2, 1/2, 1]
    r = len(B)
    A = np.zeros((r, r))
    A[1][0] = 1/2
    A[2][1] = 1/2
    A[3][2] = 1

    #Problem conditions
    t0 = 0
    tf = 1
    u0 = 1 #initial value
    N = 1000
    t_vector, h = np.linspace(t0, tf, num=N, retstep=True)

    lambdas = [-10, 1, 10]
    for l in lambdas:
        f = lambda u, t: l * u
        sol_analitica = lambda t: math.exp(l*t)
        sol_vector = np.zeros(N)
        for i in range(N):
            sol_vector[i] = sol_analitica(t_vector[i])

        #Convergence analysis
        ans1 = runge_kutta(A, B, C, f, u0, t_vector, h)
        ans2 = runge_kutta(A, B, C, f, u0, t_vector, h/2)

        err_ans1, err_ans2 = [], []
        for i in range(N):
            err_ans1.append(abs(ans1[i] - sol_vector[i]))
            err_ans2.append(abs(ans2[i] - sol_vector[i]))

        p = 2
        E_ans1 = norm_p(err_ans1, p, h)
        E_ans2 = norm_p(err_ans2, p, h/2)

        R = E_ans1 / E_ans2
        p_rk = math.log(R, 2)
        c = E_ans1 / (h**p_rk)
        print("Para lambda={0}".format(l))
        print("Metodo Runge-Kutta: c={0}, p={1}".format(c, p_rk))

        #Numeric refinement study
    return

main()
