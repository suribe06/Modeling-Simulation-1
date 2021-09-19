#Numerical solutions for EDOs

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import math

def f(t,u):
    return -10*u
    #return 10*(u-math.cos(t))-math.sin(t)

def sol_analitica(t):
    return math.exp(-10*t)
    #return math.cos(t)

def euler_explicit_method(t_vector, y0, h):
    n = len(t_vector)
    e_vector = np.zeros(n)
    e_vector[0] = y0
    for i in range(n-1):
        e_vector[i + 1] = e_vector[i] + h*f(t_vector[i], e_vector[i])
    return e_vector

def euler_implicit_method(t_vector, y0, h):
    n = len(t_vector)
    e_vector = np.zeros(n)
    e_vector[0] = y0
    for i in range(n-1):
        f_ = lambda x: x - e_vector[i] - (h * f(t_vector[i+1], x))
        #root = optimize.newton(f_, e_vector[i])
        root = optimize.fsolve((f_),(e_vector[i]))
        e_vector[i + 1] = root
    return e_vector

def trapezoid_method(t_vector, y0, h):
    n = len(t_vector)
    e_vector = np.zeros(n)
    e_vector[0] = y0
    for i in range(n-1):
        f_ = lambda x: x - e_vector[i] - ((h/2)*(f(t_vector[i], e_vector[i]) + f(t_vector[i+1], x)))
        #root = optimize.newton(f_, e_vector[i])
        root = optimize.fsolve((f_),(e_vector[i]))
        e_vector[i + 1] = root
    return e_vector

def main():
    N = 1000
    t0 = 0
    tf = 2*math.pi
    y0 = 1
    t_vector, h = np.linspace(t0, tf, num=N, retstep=True)
    n = len(t_vector)
    sol_vector = np.zeros(n)
    for i in range(len(t_vector)):
        sol_vector[i] = sol_analitica(t_vector[i])

    #Methods
    e_vector1 = euler_explicit_method(t_vector, f, y0, h)
    e_vector2 = euler_implicit_method(t_vector, y0, h)
    trap_vector = trapezoid_method(t_vector, y0, h)

    #Plot the methods
    plt.plot(t_vector, e_vector1, "cx", label = "Euler Explicit")
    plt.plot(t_vector, e_vector2, "r.", label = "Euler Implicit")
    plt.plot(t_vector, trap_vector, "b+", label = "Trapezoid")
    plt.plot(t_vector, sol_vector, "m", label = "Analytical Solution")
    plt.title("Numerical Methods for ODEs")
    plt.ylabel("f(t)")
    plt.xlabel("t")
    plt.legend()
    plt.grid()
    plt.show()
    return

#main()
