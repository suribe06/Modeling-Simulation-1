#Runge-Kutta Method for r steps

import matplotlib.pyplot as plt
from scipy import optimize
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

def implicit_runge_kutta_2(A, B, C, f, u0, t_vector, h):
    #Impicit RK method of 2 steps
    n = len(t_vector)
    r = len(B)
    u_vector = np.zeros(n) #solution vector
    u_vector[0] = u0
    for i in range(n-1):
        def equations(vars):
            #nonlinear equation system
            y1, y2 = vars
            eq1 = y1 - u_vector[i] - h * ((A[0][0]*f(y1, t_vector[i] + (C[0] * h))) + (A[0][1]*f(y2, t_vector[i] + (C[1] * h))))
            eq2 = y2 - u_vector[i] - h * ((A[1][0]*f(y1, t_vector[i] + (C[0] * h))) + (A[1][1]*f(y2, t_vector[i] + (C[1] * h))))
            return [eq1, eq2]
        y1, y2 =  optimize.fsolve(equations, (u_vector[i], u_vector[i]))
        ys = [y1, y2]
        #Update next value of u_vector
        sum = 0
        for j in range(r): #r=2
            sum += B[j] * f(ys[j], t_vector[i] + (C[j] * h))
        u_vector[i + 1] = u_vector[i] + (h * sum)
    return u_vector

def runge_kutta(A, B, C, f, u0, t_vector, h):
    #RK Method of r steps
    r = len(B)
    n = len(t_vector)
    u_vector = np.zeros(n) #solution vector
    u_vector[0] = u0
    for i in range(n-1):
        ys = [] #vector of ys
        for j in range(r):
            sum = 0
            for k in range(j):
                if A[j][k] == 0: sum += 0
                else: sum += A[j][k] * f(ys[j-1], t_vector[i] + (h*C[k]))
            y_j = u_vector[i] + (h * sum)
            ys.append(y_j)
        #Update next value of u_vector
        sum = 0
        for j in range(r):
            sum += B[j] * f(ys[j], t_vector[i] + (C[j] * h))
        u_vector[i + 1] = u_vector[i] + (h * sum)
    return u_vector

def main():
    #First Butcher table (RK4) (Explicit RK)
    A = np.array([[0,0,0,0], [1/2,0,0,0], [0,1/2,0,0], [0,0,1,0]])
    B = [1/6, 1/3, 1/3, 1/6]
    C = [0, 1/2, 1/2, 1]

    #Second Butcher table (Gauss???Legendre) (Implicit RK)
    A2 = [[1/4, (3 - 2*math.sqrt(3))/12], [(3 + 2*math.sqrt(3))/12, 1/4]]
    B2 = [1/2, 1/2]
    C2 = [(3 - math.sqrt(3))/6, (3 + math.sqrt(3))/6]

    #Problem conditions
    t0 = 0
    tf = 1
    u0 = 1 #initial value
    N = 1000
    t_vector, h = np.linspace(t0, tf, num=N, retstep=True)

    lambdas = [-10, 1, 10]
    cont = 4
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

        #Numerical refinement study
        K = 12
        E_sequence = []
        hs = []
        for i in range(1, K+1):
            h_i = h / (2**i)
            ode_approx_i = runge_kutta(A, B, C, f, u0, t_vector, h_i)
            e_i = []
            for j in range(N):
                e_i.append(abs(ode_approx_i[j] - sol_vector[j]))
            p = 2
            E_i = norm_p(e_i, p, h_i)
            E_sequence.append(E_i)
            hs.append(h_i)

        plt.clf()
        plt.plot(hs, E_sequence, "c-o", label="E Sequence for lambda={0}".format(l))
        plt.legend()
        plt.ylabel("E(h)")
        plt.xlabel("h")
        plt.grid()
        plt.savefig("NRS{0}".format(cont))
        cont += 1
    return

main()
