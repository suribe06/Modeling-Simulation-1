import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import statistics
import math

def f(t,u):
    #return -10*u
    return 10*(u-math.cos(t))-math.sin(t)

def sol_analitica(t):
    #return math.exp(-10*t)
    return math.cos(t)

def euler_explicit_method(t_vector, f, y0, h):
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
        f_ = lambda x: x - e_vector[i] + h*(f(t_vector[i+1], x))
        root = optimize.newton(f_, e_vector[i])
        e_vector[i + 1] = root
    return e_vector

def trapezoid_method(t_vector, y0, h):
    n = len(t_vector)
    e_vector = np.zeros(n)
    e_vector[0] = y0
    for i in range(n-1):
        f_ = lambda x: x - e_vector[i] + (h/2)*(f(t_vector[i], e_vector[i]) + f(t_vector[i+1], x))
        root = optimize.newton(f_, e_vector[i])
        e_vector[i + 1] = root
    return e_vector

def main():
    h = 1/999
    t0 = 0
    tf = 2*math.pi
    y0 = 1
    t_vector = np.arange(t0,tf+h,h)
    n = len(t_vector)
    sol_vector = np.zeros(n)
    for i in range(len(t_vector)):
        sol_vector[i] = sol_analitica(t_vector[i])
    error = []
    #Methods
    """
    try:
        e_vector1 = euler_explicit_method(t_vector, f, y0, h)
        plt.plot(t_vector, e_vector1, label = "Euler Explicit")
        for i in range(n):
            error.append(abs((e_vector1[i] - sol_vector[i])/sol_vector[i]))
        mre = statistics.mean(error)
        print("Mean Relative Error Euler Explicit: ",mre)
    except:
        print("El metodo euler explicito no funciona con esta ODE")
    """
    try:
        e_vector2 = euler_implicit_method(t_vector, y0, h)
        plt.plot(t_vector, e_vector2, "+", label = "Euler Implicit")
        for i in range(n):
            error.append(abs((e_vector2[i] - sol_vector[i])/sol_vector[i]))
        mre = statistics.mean(error)
        print("Mean Relative Error Euler Implicit: ",mre)
    except:
        print("El metodo euler implicito no funciona con esta ODE")

    try:
        trap_vector = trapezoid_method(t_vector, y0, h)
        plt.plot(t_vector, trap_vector, label = "Trapezoid")
        for i in range(n):
            error.append(abs((trap_vector[i] - sol_vector[i])/sol_vector[i]))
        mre = statistics.mean(error)
        print("Mean Relative Error Trapezoid: ",mre)
    except:
        print("El metodo trapezoide no funciona con esta ODE")

    #Plot the methods
    plt.plot(t_vector, sol_vector, label = "Analytical Solution")
    plt.title("Numerical Methods for ODEs")
    plt.ylabel("f(t)")
    plt.xlabel("t")
    plt.legend()
    plt.grid()
    plt.show()
    return

main()
