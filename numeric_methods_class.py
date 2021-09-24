# Created by Santiago Uribe (2021)

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import math

def norm_p(e, p, h):
    sum = 0
    for x in e: sum += abs(x)**p
    return h * (sum **(1/p))

def plot_graphics(x, y, l, xl, yl, name):
    plt.plot(x, y, "cx", label=l)
    plt.ylabel(xl)
    plt.xlabel(yl)
    plt.legend()
    plt.grid()
    plt.savefig("{0}.png".format(name), format="PNG")

class NumericMethods:
    def __init__(self, f, y0, t0, tf, N, sol_analitica=None):
        self.f = f #ode function
        self.y0 = y0 #initial value
        self.t0 = t0 #initial time
        self.tf = tf #final time
        self.N = N #number of points
        self.sol_analitica = sol_analitica
        self.t_vector, self.h = np.linspace(t0, tf, num=N, retstep=True)
        return

    def sol_analitica_vector(self):
        sol_vector = np.zeros(self.N)
        for i in range(self.N):
            sol_vector[i] = self.sol_analitica(self.t_vector[i])
        return sol_vector

    def euler_explicit(self, graphic, h=None):
        if h == None: h = self.h
        n = len(self.t_vector)
        e_vector = np.zeros(n)
        e_vector[0] = self.y0
        for i in range(n-1):
            e_vector[i + 1] = e_vector[i] + h*self.f(self.t_vector[i], e_vector[i])
        if graphic: plot_graphics(self.t_vector, e_vector, "Euler Explicit", "t", "f(t)", "EulerExplicit")
        return e_vector

    def euler_implicit(self, graphic, h=None):
        if h == None: h = self.h
        n = len(self.t_vector)
        e_vector = np.zeros(n)
        e_vector[0] = self.y0
        for i in range(n-1):
            f_ = lambda x: x - e_vector[i] - (h * self.f(self.t_vector[i+1], x))
            root = optimize.fsolve((f_),(e_vector[i]))
            e_vector[i + 1] = root
        if graphic: plot_graphics(self.t_vector, e_vector, "Euler Implicit", "t", "f(t)", "EulerImplicit")
        return e_vector

    def trapezoid(self, graphic, h=None):
        if h == None: h = self.h
        n = len(self.t_vector)
        trap_vector = np.zeros(n)
        trap_vector[0] = self.y0
        for i in range(n-1):
            f_ = lambda x: x - trap_vector[i] - ((h/2)*(self.f(self.t_vector[i], trap_vector[i]) + self.f(self.t_vector[i+1], x)))
            root = optimize.fsolve((f_),(trap_vector[i]))
            trap_vector[i + 1] = root
        if graphic: plot_graphics(self.t_vector, trap_vector, "Trapezoid", "t", "f(t)", "Trapezoid")
        return trap_vector

    def numerical_refinement_study(self, method_selector, K, norm):
        ode_approx = None
        hs = []
        E_sequence = []
        ans = False
        if method_selector == 1:
            if self.sol_analitica != None: ode_approx = self.sol_analitica_vector()
            else: ode_approx = self.euler_explicit(False)
            for i in range(1, K+1):
                h_i = self.h / (2**i)
                ode_approx_i = self.euler_explicit(False, h_i)
                e_i = []
                for j in range(self.N):
                    e_i.append(abs(ode_approx_i[j] - ode_approx[j]))
                E_i = norm_p(e_i, norm, h_i)
                E_sequence.append(E_i)
                hs.append(h_i)
            ans = True
        elif method_selector == 2:
            if self.sol_analitica != None: ode_approx = self.sol_analitica_vector()
            else: ode_approx = self.euler_implicit(False)
            for i in range(1, K+1):
                h_i = self.h / (2**i)
                ode_approx_i = self.euler_implicit(False, h_i)
                e_i = []
                for j in range(self.N):
                    e_i.append(abs(ode_approx_i[j] - ode_approx[j]))
                E_i = norm_p(e_i, norm, h_i)
                E_sequence.append(E_i)
                hs.append(h_i)
            ans = True
        elif method_selector == 3:
            if self.sol_analitica != None: ode_approx = self.sol_analitica_vector()
            else: ode_approx = self.trapezoid(False)
            for i in range(1, K+1):
                h_i = self.h / (2**i)
                ode_approx_i = self.trapezoid(False, h_i)
                e_i = []
                for j in range(self.N):
                    e_i.append(abs(ode_approx_i[j] - ode_approx[j]))
                E_i = norm_p(e_i, norm, h_i)
                E_sequence.append(E_i)
                hs.append(h_i)
            ans = True
        else:
            print("El metodo ingresado no es valido")
        if ans: plot_graphics(hs, E_sequence, "E Sequence", "E(h)", "h", "NumericalRefinementStudy")

    def convergence_analysis(self, method_selector):
        p = None
        c = None
        if method_selector == 1:
            pass
        elif method_selector == 2:
            pass
        elif method_selector == 3:
            pass
        else:
            print("El metodo ingresado no es valido")
        return c, p
