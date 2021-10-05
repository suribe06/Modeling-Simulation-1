import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import math

def f(u,t):
    return 3*u

def sol_analitica(t):
    return math.exp(3*t)

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
    ys = [] #vector of ys
    for i in range(n-1):
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

    #Second Butcher table (Gauss–Legendre) (Implicit RK)
    A2 = [[1/4, (3 - 2*math.sqrt(3))/12], [(3 + 2*math.sqrt(3))/12, 1/4]]
    B2 = [1/2, 1/2]
    C2 = [(3 - math.sqrt(3))/6, (3 + math.sqrt(3))/6]

    #Problem conditions
    t0 = 0
    tf = 1
    u0 = 1 #initial value
    N = 2
    t_vector, h = np.linspace(t0, tf, num=N, retstep=True)
    sol_vector = np.zeros(N)
    for i in range(N):
        sol_vector[i] = sol_analitica(t_vector[i])

    ans1 = runge_kutta(A, B, C, f, u0, t_vector, h)
    ans2 = implicit_runge_kutta_2(A2, B2, C2, f, u0, t_vector, h)
    print(ans1)
    plt.plot(t_vector, sol_vector, label="Sol Analitica")
    plt.plot(t_vector, ans1, label="RK Explicit")
    #plt.plot(t_vector, ans2, label="RK Implicit")
    plt.grid()
    plt.legend()
    plt.show()

main()
