w0 = 5
nw0 = 0

w1 = 5
nw1 = 0

w2 = 5
nw2 = 0

w3 = 20
nw3 = 0

w4 = 1
nw4 = 0

w5 = 1
nw5 = 0

w6 = 1
nw6 = 0

w7 = 1
nw7 = 0

w8 = 1
nw8 = 0

w9 = 1
nw9 = 0

bias = 5
nbias = 0

# Datos entrenamiento
FI = [38, 36, 37, 40, 35]
TS = [4, 2, 4, 5, 0]
DC = [5, 1, 4, 5, 0]
DR = [18, 3, 19, 20, 0]
DA = [1, 0, 1, 0, 1]
DM = [1, 0, 0, 0, 1]
DG = [1, 1, 0, 0, 1]
FN = [1, 0, 0, 0, 0]
CJ = [0, 0, 0, 0, 0]
DT = [1, 0, 0, 0, 0]

enfermo = [1, 0, 1, 1, 0]

acumError = 1
calError = 0
veces = 0
net = [0, 0, 0, 0, 0, 0, 0, 0, 0]
salida = [0, 0, 0, 0, 0, 0, 0, 0, 0]
error = [0, 0, 0, 0, 0, 0, 0, 0, 0]


def calFIRange(fi):
    if fi <= 36:
        return 0
    else:
        nFi = (fi * 20) / 38
        return 20 if nFi >= 20 else nFi


# Fase de entrenamiento
while acumError != 0:
    acumError = 0

    for i in range(5):
        net[i] = (w0 * calFIRange(FI[i])) + (w1 * TS[i]) + (w2 * DC[i]) + (w3 * DC[i]) + (w4 * DR[i]) + (w5 * DA[i]) + (
                    w6 * DM[i]) + (w7 * DG[i]) + (w8 * FN[i]) + (w8 * CJ[i]) + (w9 * DT[i]) + bias
        # funcion de activacion (escalon unitario) x>0=1  x<=0=0
        if net[i] > 0:
            salida[i] = 1
        else:
            salida[i] = 0
        # Calculo del error
        calError = enfermo[i] - salida[i]
        acumError += calError
        error[i] = calError

    # Calculo de nuevos pesos
    for i in range(9):
        if error[i] != 0:
            nw0 = w0 + (error[i] * FI[i])
            nw1 = w1 + (error[i] * TS[i])
            nw2 = w2 + (error[i] * DC[i])
            nw3 = w3 + (error[i] * DR[i])
            nw4 = w4 + (error[i] * DA[i])
            nw5 = w5 + (error[i] * DM[i])
            nw6 = w6 + (error[i] * DG[i])
            nw7 = w7 + (error[i] * FN[i])
            nw8 = w8 + (error[i] * CJ[i])
            nw9 = w9 + (error[i] * DT[i])
            nbias = bias + error[i]

    veces += 1
    w0 = nw0
    w1 = nw1
    w2 = nw2
    w3 = nw3
    w4 = nw4
    w5 = nw5
    w6 = nw6
    w7 = nw7
    w8 = nw8
    w9 = nw9
    bias = nbias

print("Veces = ", veces)
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("w4 =", w4)
print("w5 =", w5)
print("w6 =", w6)
print("w7 =", w7)
print("w8 =", w8)
print("w9 =", w9)

print("bias = ", bias)

# Fase de prueba
FI = [40, 35, 38,37.5]
TS = [5, 0, 0, 1]
DC = [5, 0, 5, 3]
DR = [5, 0, 20, 12]
DA = [0, 1, 0, 0]
DM = [0, 1, 0, 3]
DG = [0, 1, 0, 0]
FN = [1, 0, 0, 0]
CJ = [0, 0, 0, 0]
DT = [0, 0, 0, 0]

netP = [0, 0, 0, 0]
salidaP = []
for i in range(4):
    netP[i] = (w0 * calFIRange(FI[i])) + (w1 * TS[i]) + (w2 * DC[i]) + (w3 * DC[i]) + (w4 * DR[i]) + (w5 * DA[i]) + (
                w6 * DM[i]) + (w7 * DG[i]) + (w8 * FN[i]) + (w8 * CJ[i]) + (w9 * DT[i]) + bias
    if netP[i] > 0:
        salidaP.append(1)
    else:
        salidaP.append(0)
print(salidaP)
