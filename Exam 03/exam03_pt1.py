import matplotlib.pyplot as plt
import numpy as np
import math
from time import time

def sol_analitica(eigenvalues, eigenvectors, t):
    l1, l2 = eigenvalues
    v1, v2 = eigenvectors
    c1, c2 = 3/2, -2
    x1 = c1*v1[0]*math.exp(l1*t) + c2*v2[0]*math.exp(l2*t)
    x2 = c1*v1[1]*math.exp(l1*t) + c2*v2[1]*math.exp(l2*t)
    ans = [x1, x2]
    return ans

def f1(u, t):
    ans = u**2 - u**3
    return ans

def f2(u, t):
    x1, x2 = u
    dx1dt = -298*x1 + 99*x2
    dx2dt = -594*x1 + 197*x2
    ans = np.array([dx1dt, dx2dt], dtype=object)
    return ans

def adaptive_runge_kutta(A, C, B1, B2, f, u0, t0, tf, h0, e_min, e_max, h_min, h_max, max_iter):
    """
    Parameters:
    A, B1, B2, C = Butcher table
    u0 = Initial condition
    t0 = Initial time
    tf = Final time
    h0 = first step size
    e_min, e_max = range for error tolerance
    h_min, h_max = range for the step size
    """
    t_vector = [t0]
    u = [u0]#solution vector
    t = t0
    h = h0
    l = 0
    while t < tf - h and l < max_iter:
        k1 = h * f(u[-1], t)
        k2 = h * f(u[-1] + A[0][0] * k1, t + C[0] * h)
        k3 = h * f(u[-1] + A[1][0] * k1 + A[1][1] * k2, t + C[1] * h)
        k4 = h * f(u[-1] + A[2][0] * k1 + A[2][1] * k2 + A[2][2] * k3, t + C[2] * h)
        k5 = h * f(u[-1] + A[3][0] * k1 + A[3][1] * k2 + A[3][2] * k3 + A[3][3] * k4, t + C[3] * h)
        k6 = h * f(u[-1] + A[4][0] * k1 + A[4][1] * k2 + A[4][2] * k3 + A[4][3] * k4 + A[4][4] * k5, t + C[4] * h )
        ks = [k1, k2, k3, k4, k5, k6]
        #Calculate the methods
        RK4 = u[-1] + B1[0] * k1 + B1[1] * k2 + B1[2] * k3 + B1[3] * k4 + B1[4] * k5
        RK5 = u[-1] + B2[0] * k1 + B2[1] * k2 + B2[2] * k3 + B2[3] * k4 + B2[4] * k5 + B2[5] * k6
        #Calculate local truncation error
        e = None
        if u0.size == 1: e = abs(RK4 - RK5)
        else: e = np.linalg.norm(np.subtract(RK4, RK5), 2)
        if e > e_max and h > h_min: h = h/2 #reject step
        else: #accept step
            t_vector.append(t)
            #store the better answer
            u.append(RK5)
            t += h
            l += 1
            if e < e_min: h = 2*h
        #keep h in the step size range
        if h < h_min: h = h_min
        elif h > h_max: h = h_max
    return u, t_vector

#Butcher table for Runge–Kutta–Fehlberg method
A = [np.array([1/4]),
    np.array([3/32, 9/32]),
    np.array([1932/2197, -7200/2197, 7296/2197]),
    np.array([439/216, -8, 3680/513, -845/4104]),
    np.array([-8/27, 2, -3544/2565, 1859/4104, -11/40])]
B1 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
B2 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
C = [1/4, 3/8, 12/13, 1, 1/2]

#First ODE
delta = 0.00005
t0 = 0
tf = 1/delta
u0 = np.array(delta) #initial value
e_min, e_max = 1*10**-4, 1*10**-3
h_min, h_max = 1*10**-3, 5
h0 = 0.1
max_iter = 100000
t1 = time()
u, t_vec = adaptive_runge_kutta(A, C, B1, B2, f1, u0, t0, tf, h0, e_min, e_max, h_min, h_max, max_iter)
t2 = time()
print("Execution Time {0}".format(t2 - t1))
plt.plot(t_vec, u, 'm', label=' x(t) numeric solution')
plt.legend()
plt.grid()
plt.xlabel('t')
plt.show()

#Second ODE
t0 = 0
tf = 3000
#Numeric Solution
u0 = np.array([-1/2, 1/2])
e_min, e_max = 1*10**-4, 1*10**-3
h_min, h_max = 1*10**-3, 5
h0 = 0.1
max_iter = 100000
t1 = time()
u, t_vec = adaptive_runge_kutta(A, C, B1, B2, f2, u0, t0, tf, h0, e_min, e_max, h_min, h_max, max_iter)
t2 = time()
print("Execution Time {0}".format(t2 - t1))
x1, x2 = [], []
for a in u:
    x1.append(a[0])
    x2.append(a[1])
plt.clf()
plt.plot(t_vec, x1, 'm', label='x1(t) numeric')
plt.plot(t_vec, x2, 'c', label='x2(t) numeric')
plt.legend()
plt.grid()
plt.xlabel('t')
plt.show()
