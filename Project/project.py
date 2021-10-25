import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

alfa = 0.35
epsilon = 0.05
miu = 0.2
beta = 0.7
nu = 0.2
omega_0 = 0.4
omega_1 = 0.1

def omega_function(G):
    ans = None
    if G <= alfa: ans = omega_0
    elif alfa <= G and G <= (alfa + epsilon): ans = omega_0 - ((G-alfa)*(omega_0 - omega_1))/epsilon
    elif G >= (alfa + epsilon): ans = omega_1
    return ans

def ode_system(u, t):
    G, S, T = u
    dGdt = miu*S + nu*T - beta*G*T
    dSdt = beta*G*T - omega_function(G)*S - miu*S
    dTdt = omega_function(G)*S - nu*T
    ans = [dGdt, dSdt, dTdt]
    return ans

#First Simulation
G0 = 0.3
S0 = 0.2
T0 = 0.5
u0 = [G0, S0, T0]
t_max = 100
t = np.linspace(0, t_max, 1000)
#Solve de DE system
u = odeint(ode_system, u0, t)
G, S, T = u.T

plt.plot(t, G, 'b', label='G(t)')
plt.plot(t, S, 'r', label='S(t)')
plt.plot(t, T, 'g', label='T(t)')
plt.legend()
plt.grid()
plt.xlabel('Time range')
plt.ylabel('Land coverage of state variables')
plt.show()

#Second Simulation
#G0 = 0.8
#S0 = 0.1
#T0 = 0.1
