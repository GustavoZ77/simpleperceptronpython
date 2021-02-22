import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivada(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivada(x):
    return 1.0 - x ** 2


class XorBackPropNet:
    pesos = [
        [1, 2],
        [-1, -2],
        [0.5, 0.2]
    ]

    def __init__(self, fActivation):
        if fActivation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif fActivation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada

    def train(self):
        entrada = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        salida = [0, 1, 1, 0]
        y = [0, 0, 0]
        error = [0, 0, 0, 0]
        d = [0, 0, 0]
        for a in range(10000):
            for e in range(len(entrada)):
                x1 = entrada[e][0]
                x2 = entrada[e][1]

                # calculo de salidas
                for net in range(3):
                    y[net] = self.activation(self.pesos[net][0] * x1 + (self.pesos[net][1] * x2))
                # calculo de deltas
                error[e] = salida[e] - y[net]
                for net in range(3):
                    d[net] = error[e] * self.activation_prime(y[net])
                for p in range(len(self.pesos)):
                    self.pesos[p][0] = self.pesos[p][0] + (0.5 * d[p] * y[p])
                    self.pesos[p][1] = self.pesos[p][1] + (0.5 * d[p] * y[p])

        print(self.pesos)

    def test(self):
        entrada = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        y = [0, 0, 0]
        salidas = []
        for e in range(len(entrada)):
            x1 = entrada[e][0]
            x2 = entrada[e][1]

            # calculo de salidas
            for net in range(3):
                y[net] = self.activation(self.pesos[net][0] * x1 + (self.pesos[net][1] * x2))
            salidas.append(y[net])
        print(salidas)


xor = XorBackPropNet("tanh")
xor.train()
xor.test()
