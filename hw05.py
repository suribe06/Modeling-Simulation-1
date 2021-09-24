#Runge-Kutta Methods

import matplotlib.pyplot as plt
import numpy as np

def runge_kutta(A, B, C, f):
    return

def main():
    B = [1/6, 1/3, 1/3, 1/6]
    C = [0, 1/2, 1/2, 1]
    r = len(B)
    A = np.zeros((r, r))

    t0 = 0
    tf = 1
    u0 = 1 #initial value
    lambdas = [-10, 1, 10]
    for l in lambdas:
        f = lambda u: l * u
        runge_kutta(A, B, C, f)
    return

main()
