import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import math

def f(u, t):
    return u**2 - u**3

def f2(u, t):
    x1, x2 = u
    dx1dt = -298*x1 + 99*x2
    dx2dt = -594*x1 + 197*x2
    ans = [dx1dt, dx2dt]
    return ans

def runge_kutta_adaptative(A, B, C, f, u0, t0, tf, h0, e_min, e_max, h_min, h_max, max_iter):
    """
    Parameters:
    A, B, C = Butcher table
    u0 = Initial condition
    t0 = Initial time
    tf = Final time
    h = first step size
    e_min, e_max = range for error tolerance
    h_min, h_max = range for the step size
    """
    #RK Method of r steps
    r = len(B)
    t_vector = [t0]
    u_vector = [u0] #numeric solution
    u_vector_RK4 = []
    u_vector_RK5 = []
    u_vector_RK5.append(u0)
    u_vector_RK4.append(u0)
    t = t0
    h = h0
    k = 0
    while t < tf - h and k < max_iter:
        ys_RK4 = []
        ys_RK5 = []
        for j in range(r):
            if j <= 3: #for RK-4 and RK-5
                sum = 0
                for k in range(j):
                    if A[j][k] != 0: sum += A[j][k] * f(ys_RK4[j-1], t + (h*C[k]))
                y_j = u_vector_RK4[-1] + (h * sum)
                ys_RK4.append(y_j)
                ys_RK5.append(y_j)
            elif j > 3: #only for RK-5
                sum = 0
                for k in range(j):
                    if A[j][k] != 0: sum += A[j][k] * f(ys_RK5[j-1], t + (h*C[k]))
                y_j = u_vector_RK5[-1] + (h * sum)
                ys_RK5.append(y_j)
        #RK-4
        sum = 0
        for j in range(r-1):
            sum += B[j] * f(ys_RK4[j], t + (C[j] * h))
        u_vector_RK4.append(u_vector_RK4[-1] + (h * sum))
        #RK-5
        sum = 0
        for j in range(r):
            sum += B[j] * f(ys_RK5[j], t + (C[j] * h))
        u_vector_RK5.append(u_vector_RK5[-1] + (h * sum))

        #Choice of step size
        e = abs(u_vector_RK4[-1] - u_vector_RK5[-1]) #Calulate local truncation error
        if e > e_max and h > h_min: h=h/2 #reject step
        else: #accept step
            t_vector.append(t)
            u_vector.append(u_vector_RK5[-1])#store the better answer
            t = t + h
            k = k + 1
            if e < e_min: h = 2*h

    return u_vector, t_vector

A = [np.array([1/5]),
    np.array([3/40, 9/40]),
    np.array([44/45, -56/15, 32/9]),
    np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
    np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656])]

B = [25/216, 0, 1408/2565, 2197/4104, -1/5]
C = [1/4, 3/8, 12/13, 1, 1/2]

#First ODE
delta = 0.01
t0 = 0
tf = 1/delta
u0 = delta #initial value
e_min, e_max = 1*10**-4, 1*10**-3
h_min, h_max = 1*10**-3, 2
u_vector, t_vector = runge_kutta_adaptative(A, B, C, f, u0, t0, tf, 0.1, e_min, e_max, h_min, h_max, 500)
plt.plot(t_vector, u_vector, 'm', label='numeric solution')
plt.legend()
plt.grid()
plt.xlabel('t')
plt.show()

#Second ODE System
t0 = 0
tf = 20000
u0 = [-1/2, 1/2]
