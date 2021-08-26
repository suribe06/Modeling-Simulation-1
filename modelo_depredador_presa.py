import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

K = 10000
r1 = 0.1
a1 = 0.005
r2 = 0.04
a2 = 0.00004

#Initial conditions
P0, D0 = 2000, 10
initial_conditions = [P0, D0]

# The SIR model differential equations.
def de_model(y, t, r1, a1, r2, a2, K):
    P, D = y
    dPdt = r1*P*(1-P/K) - a1*P*D
    dDdt = a2*P*D - r2*D
    return dPdt, dDdt

def simulation1(t_max):
    t = np.linspace(0, t_max, 1000)
    #Solve de DE system
    sol = odeint(de_model, initial_conditions, t, args=(r1, a1, r2, a2, K))
    P, D = sol.T
    #Plot the curves for P(t) and D(t)
    plot_model(t, P, D)
    return

def plot_model(t, P, D):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax2=fig.add_subplot(111, frame_on=False)
    ax=fig.add_subplot(111)
    ax2=fig.add_subplot(111, frame_on=False)
    ax2.yaxis.tick_right()
    ax.plot(t, P,'b', label='P(t): Presa (conejo)')
    ax2.plot(t, D,'r', label='D(t): Depredador (zorro)')
    ax.set_ylabel("Poblacion Conejos")
    ax2.set_ylabel("Poblacion Zorros")
    ax2.yaxis.set_label_position('right')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid()
    plt.xlabel('Time (meses)')
    plt.title('Modelo Depredador-Presa')
    plt.show()

t_max = 250
simulation1(t_max)
