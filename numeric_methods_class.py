# Created by Santiago Uribe (2021)

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import math

def norm_p(e, p, h):
    ans = None
    if p == np.inf:
        ans = max(e)
    else:
        sum = 0
        for x in e: sum += abs(x)**p
        ans = h * (sum **(1/p))
    return ans

def plot_graphics(x, y, l, xl, yl, name):
    plt.clf()
    plt.plot(x, y, "cx", label=l)
    plt.ylabel(xl)
    plt.xlabel(yl)
    plt.legend()
    plt.grid()
    plt.savefig("{0}.png".format(name), format="PNG")

class NumericMethods:
    def __init__(self, f, y0, t0, tf, N, sol_analitica=None):
        #Constructor
        self.f = f #ode function
        self.y0 = y0 #initial value
        self.t0 = t0 #initial time
        self.tf = tf #final time
        self.N = N #number of points
        self.sol_analitica = sol_analitica
        self.t_vector, self.h = np.linspace(t0, tf, num=N, retstep=True)
        return

    def get_sol_analitica_vector(self):
        sol_vector = np.zeros(self.N)
        for i in range(self.N):
            sol_vector[i] = self.sol_analitica(self.t_vector[i])
        return sol_vector

    def get_t_vector(self):
        return self.t_vector

    def euler_explicit(self, graphic, h=None):
        #Explicit Euler Method
        if h == None: h = self.h
        n = len(self.t_vector)
        e_vector = np.zeros(n)
        e_vector[0] = self.y0
        for i in range(n-1):
            e_vector[i + 1] = e_vector[i] + h*self.f(self.t_vector[i], e_vector[i])
        if graphic: plot_graphics(self.t_vector, e_vector, "Euler Explicit", "t", "f(t)", "EulerExplicit")
        return e_vector

    def euler_implicit(self, graphic, h=None):
        #Implicit Euler Method
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
        #Trapezoid Method
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

    def runge_kutta_implicit_2s(self, A, B, C, graphic, h=None):
        #Implicit Runge-Kuta Method of 2 Steps
        if h == None: h = self.h
        r = len(B)
        n = len(self.t_vector)
        u_vector = np.zeros(n)
        u_vector[0] = self.y0
        for i in range(n-1):
            def equations(vars):
                #nonlinear equation system
                y1, y2 = vars
                eq1 = y1 - u_vector[i] - h * ((A[0][0]*f(y1, self.t_vector[i] + (C[0] * h))) + (A[0][1]*self.f(y2, self.t_vector[i] + (C[1] * h))))
                eq2 = y2 - u_vector[i] - h * ((A[1][0]*f(y1, self.t_vector[i] + (C[0] * h))) + (A[1][1]*self.f(y2, self.t_vector[i] + (C[1] * h))))
                return [eq1, eq2]
            y1, y2 =  optimize.fsolve(equations, (u_vector[i], u_vector[i]))
            ys = [y1, y2]
            sum = 0
            for j in range(r): #r=2
                sum += B[j] * self.f(ys[j], self.t_vector[i] + (C[j] * h))
            u_vector[i + 1] = u_vector[i] + (h * sum)
        if graphic: plot_graphics(self.t_vector, u_vector, "Implicit Runge-Kuta", "t", "f(t)", "ImplicitRK")
        return u_vector

    def runge_kutta_explicit_rs(self, A, B, C, graphic, h=None):
        #Explicit Runge-Kuta Method of r Steps
        if h == None: h = self.h
        r = len(B)
        n = len(self.t_vector)
        u_vector = np.zeros(n)
        u_vector[0] = self.y0
        ys = [] #vector of ys
        for i in range(n-1):
            for j in range(r):
                sum = 0
                for k in range(j+1):
                    if A[j][k] == 0: sum += 0
                    else: #A[j][k] != 0
                        if k < j: #Rung-Kutta Explicito
                            sum += A[j][k] * self.f(ys[j-1], self.t_vector[i] + (h*C[k]))
                        else: # k >= j Runge-Kutta Implicito
                            print("La matriz A ingresada corresponde a RK implicito")
                y_i = u_vector[i] + (h * sum)
                ys.append(y_i)
            sum = 0
            for j in range(r):
                sum += B[j] * self.f(ys[j], self.t_vector[i] + (C[j] * h))
            u_vector[i + 1] = u_vector[i] + (h * sum)
        if graphic: plot_graphics(self.t_vector, u_vector, "Explicit Runge-Kuta", "t", "f(t)", "ExplicitRK")
        return u_vector

    def numerical_refinement_study(self, method_selector, K, norm):
        ode_approx = None
        hs = []
        E_sequence = []
        ans = False
        if method_selector == 0: #Explicit Euler
            if self.sol_analitica != None: ode_approx = self.get_sol_analitica_vector()
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
        elif method_selector == 1: #Implicit Euler
            if self.sol_analitica != None: ode_approx = self.get_sol_analitica_vector()
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
        elif method_selector == 2: #Trapezoid Method
            if self.sol_analitica != None: ode_approx = self.get_sol_analitica_vector()
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
        elif method_selector == 3: #Implicit RK
            pass
        elif method_selector == 4: #Explicit RK
            pass
        else:
            print("El metodo ingresado no es valido")
        if ans: plot_graphics(hs, E_sequence, "E Sequence", "E(h)", "h", "NumericalRefinementStudy")

    def convergence_analysis(self, method_selector, norm):
        p = None
        c = None
        reference_grid = None
        approx1 = None
        approx2 = None
        ans = False
        if self.sol_analitica != None: reference_grid = self.get_sol_analitica_vector()

        if method_selector == 0: #Explicit Euler
            if self.sol_analitica == None: reference_grid = self.euler_explicit(False)
            approx1 = self.euler_explicit(False, self.h/2)
            approx2 = self.euler_explicit(False, self.h/4)
            ans = True
        elif method_selector == 1: #Implicit Euler
            if self.sol_analitica == None: reference_grid = self.euler_implicit(False)
            approx1 = self.euler_implicit(False, self.h/2)
            approx2 = self.euler_implicit(False, self.h/4)
            ans = True
        elif method_selector == 2: #Trapezoid Method
            if self.sol_analitica == None: reference_grid = self.trapezoid(False)
            approx1 = self.trapezoid(False, self.h/2)
            approx2 = self.trapezoid(False, self.h/4)
            ans = True
        elif method_selector == 3: #Implicit RK
            pass
        elif method_selector == 4: #Explicit RK
            pass
        else:
            print("El metodo ingresado no es valido")

        if ans:
            err_ap1, err_ap2 = [], []
            for i in range(self.N):
                err_ap1.append(abs(approx1[i] - reference_grid[i]))
                err_ap2.append(abs(approx2[i] - reference_grid[i]))
            E_ap1 = norm_p(err_ap1, norm, self.h/2)
            E_ap2 = norm_p(err_ap2, norm, self.h/4)
            R = E_ap1 / E_ap2
            p = math.log(R, 2)
            c = E_ap1 / (self.h**p)
        return c, p
