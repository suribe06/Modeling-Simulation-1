#Numerical estimation of error in ODEs

from hw02 import euler_explicit_method, euler_implicit_method, trapezoid_method, sol_analitica, f
import numpy as np
import math

def norm_p(e, p, h):
    sum = 0
    for x in e: sum += abs(x)**p
    return h * (sum **(1/p))

def norm_inf(e):
    return max(e)

def main():
    N = 100
    t0 = 0
    tf = 2*math.pi
    y0 = 1
    t_vector, h = np.linspace(t0, tf, num=N, retstep=True)
    N1 = (2*N)-1
    t_vector2, h2 = np.linspace(t0, tf, num=N1, retstep=True)
    sol_vector = np.zeros(N)
    for i in range(N):
        sol_vector[i] = sol_analitica(t_vector[i])

    #Methods with h
    eu_expl1 = euler_explicit_method(t_vector, y0, h)
    eu_impl1 = euler_implicit_method(t_vector, y0, h)
    trap1 = trapezoid_method(t_vector, y0, h)

    #Methods with h/2
    eu_expl2 = euler_explicit_method(t_vector, y0, h2)
    eu_impl2 = euler_implicit_method(t_vector, y0, h2)
    trap2 = trapezoid_method(t_vector, y0, h2)

    #Numeric error
    e_eu_expl1, e_eu_impl1, e_trap1 = [], [], []
    for i in range(N):
        e_eu_expl1.append(abs(eu_expl1[i] - sol_vector[i]))
        e_eu_impl1.append(abs(eu_impl1[i] - sol_vector[i]))
        e_trap1.append(abs(trap1[i] - sol_vector[i]))

    e_eu_expl2, e_eu_impl2, e_trap2 = [], [], []
    for i in range(N):
        e_eu_expl2.append(abs(eu_expl2[i] - sol_vector[i]))
        e_eu_impl2.append(abs(eu_impl2[i] - sol_vector[i]))
        e_trap2.append(abs(trap2[i] - sol_vector[i]))

    p = 2
    E_eu_expl1 = norm_p(e_eu_expl1, p, h)
    E_eu_impl1 = norm_p(e_eu_impl1, p, h)
    E_trap1 = norm_p(e_trap1, p, h)
    E_eu_expl2 = norm_p(e_eu_expl2, p, h2)
    E_eu_impl2 = norm_p(e_eu_impl2, p, h2)
    E_trap2 = norm_p(e_trap2, p, h2)

    R_eu_expl = E_eu_expl1 / E_eu_expl2
    R_eu_impl = E_eu_impl1 / E_eu_impl2
    R_trap = E_trap1 / E_trap2

    p_eu_expl = math.log(R_eu_expl, 2)
    p_eu_impl = math.log(R_eu_impl, 2)
    p_trap = math.log(R_trap, 2)

    c_eu_expl = E_eu_expl1 / (h**p_eu_expl)
    c_eu_impl = E_eu_impl1 / (h**p_eu_impl)
    c_trap = E_trap1 / (h**p_trap)

    print("Metodo Euler Explicito: c={0}, p={1}".format(c_eu_expl, p_eu_expl))
    print("Metodo Euler Implicito: c={0}, p={1}".format(c_eu_impl, p_eu_impl))
    print("Metodo Trapezoide: c={0}, p={1}".format(c_trap, p_trap))
    return

main()
