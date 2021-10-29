import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numeric_methods import euler_explicit_method, euler_implicit_method, trapezoid_method
from numeric_methods import runge_kutta_implicit_2s, runge_kutta_explicit_rs

#Parameters for omega function
alfa = 0.35
epsilon = 0.05
omega_0 = 0.4
omega_1 = 0.1
#Parameters of ODE system
miu = 0.2
beta = 0.7
nu = 0.1

def plot_graphics(t_vector, G, S, T, name):
    plt.clf()
    plt.plot(t_vector, G, 'g', label='G(t)')
    plt.plot(t_vector, S, 'r', label='S(t)')
    plt.plot(t_vector, T, 'b', label='T(t)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.xlabel('Time range')
    plt.ylabel('Land coverage of state variables')
    plt.savefig("{0}.png".format(name))

def omega_function(G):
    ans = None
    if G <= alfa: ans = omega_0
    elif alfa <= G and G <= (alfa + epsilon): ans = omega_0 - ((G-alfa)*(omega_0 - omega_1))/epsilon
    elif G >= (alfa + epsilon): ans = omega_1
    return ans

def ode_system(u, t):
    G, S, T = u
    omega = omega_function(G)
    dGdt = miu*S + nu*T - beta*G*T
    dSdt = beta*G*T - omega*S - miu*S
    dTdt = omega*S - nu*T
    ans = [dGdt, dSdt, dTdt]
    return ans

#First Simulation
G0 = 0.8
S0 = 0.1
T0 = 0.1
u0 = [G0, S0, T0]
t0 = 0
tf = 100
N = 1000
t_vector, h = np.linspace(t0, tf, num=N, retstep=True)

#odeint method
u = odeint(ode_system, u0, t_vector)
G, S, T = u.T
plot_graphics(t_vector, G, S, T, "odeint()_solution")

#euler explicit method
G, S, T = euler_explicit_method(ode_system, t_vector, u0, h)
plot_graphics(t_vector, G, S, T, "euler_explicit_solution")

#euler implicit method
G, S, T = euler_implicit_method(ode_system, t_vector, u0, h)
plot_graphics(t_vector, G, S, T, "euler_implicit_solution")

#trapezoid method
G, S, T = trapezoid_method(ode_system, t_vector, u0, h)
plot_graphics(t_vector, G, S, T, "trapezoid_solution")
