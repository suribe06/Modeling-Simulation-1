import math
import numpy as np
import matplotlib.pyplot as plt
from ODE_system_solver import euler_explicit_method, euler_implicit_method, trapezoid_method

k1 = 3
k2 = 2

def plot_graphics(t_vector, x1t, x2t, x3t, x1, x2, x3, name):
    plt.clf()
    #Analytic Solution
    plt.plot(t_vector, x1t, 'r', label='analytic solution x1(t)')
    plt.plot(t_vector, x2t, 'b', label='analytic solution x2(t)')
    plt.plot(t_vector, x3t, 'g', label='analytic solution x3(t)')
    #Numeric Solution
    plt.plot(t_vector, x1, 'm', label='numeric solution x1(t)')
    plt.plot(t_vector, x2, 'c', label='numeric solution x2(t)')
    plt.plot(t_vector, x3, 'y', label='numeric solution x3(t)')
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.savefig("{0}.png".format(name))

def sol_analitica(eigenvalues, eigenvectors, t):
    l1, l2, l3 = eigenvalues
    v1, v2, v3 = eigenvectors
    x1 = v1[0]*math.exp(l1*t) + v2[0]*math.exp(l2*t) + v3[0]*math.exp(l3*t)
    x2 = v1[1]*math.exp(l1*t) + v2[1]*math.exp(l2*t) + v3[1]*math.exp(l3*t)
    x3 = v1[2]*math.exp(l1*t) + v2[2]*math.exp(l2*t) + v3[2]*math.exp(l3*t)
    ans = [x1, x2, x3]
    return ans

def ode_system(u, t):
    x1, x2, x3 = u
    dx1dt = -k1*x1
    dx2dt = k1*x1 - k2*x2
    dx3dt = k2*x2
    ans = [dx1dt, dx2dt, dx3dt]
    return ans

t0 = 0
tf = 10
N = 100
t_vector, h = np.linspace(t0, tf, num=N, retstep=True)
print("delta t = {0}".format(h))

l1 = 0
l2 = -2
l3 = -3
eigenvalues = [l1, l2, l3]
v1 = [0,0,1]
v2 = [0,-1,1]
v3 = [1,-3,2]
eigenvectors = [v1, v2, v3]

#initial conditions
u0 = [v3[0], v2[1]+v3[1], v1[2]+v2[2]+v3[2]]

x1t, x2t, x3t = [], [], []
for t in t_vector:
    ans = sol_analitica(eigenvalues, eigenvectors, t)
    x1t.append(ans[0])
    x2t.append(ans[1])
    x3t.append(ans[2])

x1, x2, x3 = euler_explicit_method(ode_system, t_vector, u0, h)
plot_graphics(t_vector, x1t, x2t, x3t, x1, x2, x3, "euler_explicit.png")

x1, x2, x3 = euler_implicit_method(ode_system, t_vector, u0, h)
plot_graphics(t_vector, x1t, x2t, x3t, x1, x2, x3, "euler_implicit.png")

x1, x2, x3 = trapezoid_method(ode_system, t_vector, u0, h)
plot_graphics(t_vector, x1t, x2t, x3t, x1, x2, x3, "trapezoid.png")
