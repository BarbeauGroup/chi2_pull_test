import numpy as np

# 3nu best fits NO IC24 + SK atm
th12 = 33.68 # in degrees
th13 = 8.56 # in degrees
th23 = 43.3 # in degrees
delta13 = 212.0 # in degrees
dm2_21 = 0 #7.49e-5
dm2_31 = 0 #2.513e-3
dm2_32 = dm2_31 - dm2_21

th14 = 20.0 # in degrees
th24 = 20.0 # in degrees
th34 = 3.0 # in degrees
delta14 = 180.0 # in degrees
delta34 = 0.0 # in degrees
dm2_41 = 1.0
dm2_42 = dm2_41 - dm2_21
dm2_43 = dm2_41 - dm2_31

dm2 = np.asarray([[0, -dm2_21, -dm2_31, -dm2_41], [dm2_21, 0, -dm2_32, -dm2_42], [dm2_31, dm2_32, 0, -dm2_43], [dm2_41, dm2_42, dm2_43, 0]])

R12 = np.asarray([[np.cos(np.radians(th12)), np.sin(np.radians(th12)), 0, 0], [-np.sin(np.radians(th12)), np.cos(np.radians(th12)), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
R13 = np.asarray([[np.cos(np.radians(th13)), 0, np.sin(np.radians(th13))*np.exp(-np.radians(delta13)*1j), 0], [0, 1, 0, 0], [-np.sin(np.radians(th13))*np.exp(np.radians(delta13)*1j), 0, np.cos(np.radians(th13)), 0], [0, 0, 0, 1]])
R23 = np.asarray([[1, 0, 0, 0], [0, np.cos(np.radians(th23)), np.sin(np.radians(th23)), 0], [0, -np.sin(np.radians(th23)), np.cos(np.radians(th23)), 0], [0, 0, 0, 1]])
R14 = np.asarray([[np.cos(np.radians(th14)), 0, 0, np.sin(np.radians(th14))*np.exp(-np.radians(delta14)*1j)], [0, 1, 0, 0], [0, 0, 1, 0], [-np.sin(np.radians(th14))*np.exp(np.radians(delta14)*1j), 0, 0, np.cos(np.radians(th14))]])
R24 = np.asarray([[1, 0, 0, 0], [0, np.cos(np.radians(th24)), 0, np.sin(np.radians(th24))], [0, 0, 1, 0], [0, -np.sin(np.radians(th24)), 0, np.cos(np.radians(th24))]])
R34 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(np.radians(th34)), np.sin(np.radians(th34))*np.exp(-np.radians(delta34)*1j)], [0, 0, -np.sin(np.radians(th34))*np.exp(np.radians(delta34)*1j), np.cos(np.radians(th34))]])

U = R34 @ R24 @ R14 @ R23 @ R13 @ R12

# PMNS matrix
# print(U) # First  index is row (flavor), second index is column (mass)

print(np.isclose(U @ np.conj(U.T), np.eye(4)))

def Pab(a, b, L, E):
    P = 0
    Impart = 0
    for i in range(4):
        for j in range(4):
            Phi = 1.27*L/E*dm2[i][j]
            Lambda = U[a][i]*np.conj(U[b][i])*np.conj(U[a][j])*U[b][j]
            P += Lambda*np.exp(-2j*Phi)

            if i == 3 and i > j:
                print(Lambda)
                Impart += Lambda

    return P, Impart

print(Pab(0, 1, 20, 30)) # P(e, mu) 20m 30MeV
# print(Pab(1, 0, 20, 30)) # P(mu, e) 20m 30MeV

Ue4_2 = np.square(np.sin(th14))
Umu4_2 = np.square(np.cos(th14)*np.sin(th24))

print("Ue4_2: ", U[0][3]*np.conj(U[0][3]))
print("Umu4_2: ", U[1][3]*np.conj(U[1][3]))