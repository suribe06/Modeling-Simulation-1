import numpy as np
import matplotlib.pyplot as plt

#Ejercicio 1
def punto1():
    struct = [('Especie', (np.str_, 30)), ('Poblacion', np.int32), ('Masa', np.float64)]
    values = [("Bowhead whale", 9000, 60), ("Blue whale", 20000, 120), ("Fin whale", 100000, 70),
            ("Humpback whale", 80000, 30), ("Gray whale", 26000, 35), ("Atlantic white-sided dolphin", 250000, 0.235),
            ("Pacific white-sided dolphin", 1000000, 0.15), ("Killer whale", 100000, 4.5), ("Narwhal", 25000, 1.5),
            ("Beluga", 100000, 1.5), ("Sperm whale", 2000000, 50), ("Baiji", 13, 0.13),
            ("North Atlantic right whale", 300, 75), ("North Pacific right whale", 200, 80), ("Southern right whale", 7000, 70)]
    A = np.array(values, dtype=struct)
    A.sort(order=['Poblacion', 'Masa'])
    print(A)

    #Entrada nueva ("Bryde whale", 100000, 25)
    return

#Ejercicio 2
def g(x, miu, sigma):
    ans = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-miu)**2/(2*sigma**2))
    return ans

def punto2():
    x_values = np.linspace(-10, 10, 1000)
    y_values = []
    sigma_values = [0.5, 1, 1.5]
    miu = 0

    for sigma in sigma_values:
        y = []
        for x in x_values:
            y.append(g(x, miu, sigma))
        y_values.append(y)

    print(sum(y_values[0]))
    for i in range(len(y_values)):
        plt.plot(x_values, y_values[i], label="g(x) con miu={0} y sigma={1}".format(miu,sigma_values[i]))

    plt.title("Función Gaussiana Normalizada")
    plt.legend()
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.show()
    return

def main():
    punto1()
    punto2()
    return

main()
