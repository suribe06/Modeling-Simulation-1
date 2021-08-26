import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

N = 100 #Total population
miu = 0.7 #transmission rate
beta = 0.1 #recovery rate

#Initial conditions
I0, R0 = 1, 0
S0 = N - I0 - R0
initial_conditions = [S0, I0, R0]

# The SIR model differential equations.
def de_model(y, t, N, miu, beta):
    S, I, R = y
    dSdt = -miu*S*I/N
    dIdt = miu*S*I/N - beta*I
    dRdt = beta*I
    return dSdt, dIdt, dRdt

def simulation1(t_max):
    t = np.linspace(0, t_max, 1000)
    #Solve de DE system
    sol = odeint(de_model, initial_conditions, t, args=(N, miu, beta))
    S, I, R = sol.T
    #Plot the curves for S(t), I(t) and R(t)
    plot_model(t, S, I, R)
    return

def simulation2(t_max):
    S = [S0]
    I = [I0]
    R = [R0]
    for t in range(1,t_max):
        S.append(S[t-1] + (-miu*S[t-1]*I[t-1]/N))
        I.append(I[t-1] + (miu*S[t-1]*I[t-1]/N - beta*I[t-1]))
        R.append(R[t-1] + (beta * I[t-1]))
    t = np.linspace(0, t_max, t_max)
    #Plot the curves for S(t), I(t) and R(t)
    plot_model(t, S, I, R)
    return

def plot_model(t, S, I, R):
    plt.plot(t, S, 'b', label='S(t): Susceptible')
    plt.plot(t, I, 'r', label='I(t): Infected')
    plt.plot(t, R, 'g', label='R(t): Recovered')
    plt.legend()
    plt.grid()
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('Model SIR')
    plt.show()

t_max = 60
simulation1(t_max)
simulation2(t_max)
