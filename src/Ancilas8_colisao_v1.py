from qutip import *
from math import *
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import linalg as LA
from numpy.linalg import norm
import scipy.constants as constant
# import matplotlib.pyplot as plt

tin = time.time()


##########  QFI com 8 camadas de Ancillas


###############################################################################
#################################  Parameters  ################################
# hp = 6.62607015E-34 #m2 kg / s
# kb = 1.380649E-23   #m2 kg s-2 K-1

gamma = 1.0  # System decay rate
gammaE = 1.0  # Environment decay rate


OmegaS = 1.0#1E10*(T/2)  # System: energy level splitting
Omega1 = 1.0
Omega2 = 1.0
Omega3 = 1.0
Omega4 = 1.0
Omega5 = 1.0
Omega6 = 1.0
Omega7 = 1.0
Omega8 = 1.0
g = 1.0  # Environment: energy level splitting
J = 1.0       # System-Environment coupling

# raz_To= (constant.k*T)/(constant.h*Omega)
# raz_To= (kb*T)/((hp/2*pi)*Omega)

# print(raz_To)

tT = 100
TempMin = 0.001
TempMax = 2.0
Temp = linspace(TempMin,TempMax,tT) # Temperature

dtT = tT - 1
ddTemp = linspace(TempMin,TempMax,dtT) # Temperature

dTemp = Temp[1] - Temp[0]

tp = 100 # Step
tSA1 = linspace(0.00001,pi/100,tp)  # Time S-A
# tSA1 = linspace(0.00001,pi/100,tp)  # Time S-A1
tA1A2 = linspace(0.00001,pi/2,tp)  # Time A1-A2
tA2A3 = linspace(0.00001,pi/2,tp)  # Time A2-A3
tA3A4 = linspace(0.00001,pi/2,tp)  # Time A3-A4
tA4A5 = linspace(0.00001,pi/2,tp)  # Time A4-A5
tA5A6 = linspace(0.00001,pi/2,tp)  # Time A5-A6
tA6A7 = linspace(0.00001,pi/2,tp)  # Time A6-A7
tA7A8 = linspace(0.00001,pi/2,tp)  # Time A7-A8
tSE = linspace(0.00001,0.1,tp)  # Time S-E
# tSE = linspace(0.00001,0.1,tp)  # Time S-E

td = linspace(0.0,5.0,tp-1)
# dt = 5/tp
dtSA1 = tSA1[1]-tSA1[0]
dtA1A2 = tA1A2[1]-tA1A2[0]

n = 30
NN = range(0, n)
Tp = len(Temp)

###############################################################################

# POVM Parameter
theta = linspace(0,pi,10)
phi = linspace(0,2*pi,20)

dTheta = theta[1]-theta[0]
dPhi = phi[1]-phi[0]

#################

nthermo = np.zeros((len(Temp)))

FthGab = np.zeros((len(Temp)))

p1dA1 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA1 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA1 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA1 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A1 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A1 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

#################
p1dA2 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA2 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA2 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA2 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A2 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A2 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

#################
p1dA3 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA3 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA3 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA3 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A3 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A3 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

#################
p1dA4 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA4 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA4 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA4 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A4 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A4 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

#################
p1dA5 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA5 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA5 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA5 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A5 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A5 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

#################
p1dA6 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA6 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA6 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA6 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A6 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A6 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

#################
p1dA7 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA7 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA7 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA7 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A7 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A7 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

#################
p1dA8 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA8 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA8 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA8 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A8 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A8 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

#################
F1A1 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
F2A1 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

FxA1 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

QFI_tA1 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta)))
# QFI_t = np.zeros((len(Temp)-1,len(range(0, n)),len(phi)))

QFIA1 = np.zeros((len(Temp)-1,len(range(0, n))))

#################
F1A2 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
F2A2 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

FxA2 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

QFI_tA2 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta)))
# QFI_t = np.zeros((len(Temp)-1,len(range(0, n)),len(phi)))

QFIA2 = np.zeros((len(Temp)-1,len(range(0, n))))

#################
F1A3 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
F2A3 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

FxA3 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

QFI_tA3 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta)))
# QFI_t = np.zeros((len(Temp)-1,len(range(0, n)),len(phi)))

QFIA3 = np.zeros((len(Temp)-1,len(range(0, n))))

#################
F1A4 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
F2A4 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

FxA4 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

QFI_tA4 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta)))
# QFI_t = np.zeros((len(Temp)-1,len(range(0, n)),len(phi)))

QFIA4 = np.zeros((len(Temp)-1,len(range(0, n))))

#################
F1A5 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
F2A5 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

FxA5 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

QFI_tA5 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta)))
# QFI_t = np.zeros((len(Temp)-1,len(range(0, n)),len(phi)))

QFIA5 = np.zeros((len(Temp)-1,len(range(0, n))))

#################
F1A6 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
F2A6 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

FxA6 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

QFI_tA6 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta)))
# QFI_t = np.zeros((len(Temp)-1,len(range(0, n)),len(phi)))

QFIA6 = np.zeros((len(Temp)-1,len(range(0, n))))

#################
F1A7 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
F2A7 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

FxA7 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

QFI_tA7 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta)))
# QFI_t = np.zeros((len(Temp)-1,len(range(0, n)),len(phi)))

QFIA7 = np.zeros((len(Temp)-1,len(range(0, n))))

#################
F1A8 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
F2A8 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

FxA8 =  np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

QFI_tA8 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta)))
# QFI_t = np.zeros((len(Temp)-1,len(range(0, n)),len(phi)))

QFIA8 = np.zeros((len(Temp)-1,len(range(0, n))))

PA = np.zeros(len(NN))
###############################################################################
#################################  Operadores  ################################

N=9

# System Alone:
sm= sigmap()
sp= sigmam()
sx= sigmax()
sy= sigmay()
sz= sigmaz()

# System + 8 Ancillas:
# System
Sm = tensor(sigmap(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Sp = tensor(sigmam(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Sx = tensor(sigmax(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Sy = tensor(sigmay(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Sz = tensor(sigmaz(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))

# Ancilla 1:
Am1 = tensor(qeye(2),sigmap(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ap1 = tensor(qeye(2),sigmam(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ax1 = tensor(qeye(2),sigmax(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ay1 = tensor(qeye(2),sigmay(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Az1 = tensor(qeye(2),sigmaz(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))

# Ancilla 2:
Am2 = tensor(qeye(2),qeye(2),sigmap(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ap2 = tensor(qeye(2),qeye(2),sigmam(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ax2 = tensor(qeye(2),qeye(2),sigmax(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ay2 = tensor(qeye(2),qeye(2),sigmay(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Az2 = tensor(qeye(2),qeye(2),sigmaz(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))

# Ancilla 3:
Am3 = tensor(qeye(2),qeye(2),qeye(2),sigmap(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ap3 = tensor(qeye(2),qeye(2),qeye(2),sigmam(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ax3 = tensor(qeye(2),qeye(2),qeye(2),sigmax(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Ay3 = tensor(qeye(2),qeye(2),qeye(2),sigmay(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))
Az3 = tensor(qeye(2),qeye(2),qeye(2),sigmaz(),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2))

# Ancilla 4:
Am4 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),sigmap(),qeye(2),qeye(2),qeye(2),qeye(2))
Ap4 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),sigmam(),qeye(2),qeye(2),qeye(2),qeye(2))
Ax4 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),sigmax(),qeye(2),qeye(2),qeye(2),qeye(2))
Ay4 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),sigmay(),qeye(2),qeye(2),qeye(2),qeye(2))
Az4 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),sigmaz(),qeye(2),qeye(2),qeye(2),qeye(2))

# Ancilla 5:
Am5 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmap(),qeye(2),qeye(2),qeye(2))
Ap5 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmam(),qeye(2),qeye(2),qeye(2))
Ax5 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmax(),qeye(2),qeye(2),qeye(2))
Ay5 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmay(),qeye(2),qeye(2),qeye(2))
Az5 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmaz(),qeye(2),qeye(2),qeye(2))

# Ancilla 6:
Am6 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmap(),qeye(2),qeye(2))
Ap6 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmam(),qeye(2),qeye(2))
Ax6 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmax(),qeye(2),qeye(2))
Ay6 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmay(),qeye(2),qeye(2))
Az6 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmaz(),qeye(2),qeye(2))

# Ancilla 7:
Am7 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmap(),qeye(2))
Ap7 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmam(),qeye(2))
Ax7 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmax(),qeye(2))
Ay7 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmay(),qeye(2))
Az7 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmaz(),qeye(2))

# Ancilla 8:
Am8 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmap())
Ap8 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmam())
Ax8 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmax())
Ay8 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmay())
Az8 = tensor(qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),qeye(2),sigmaz())
###############################################################################
# TempMax = 0.3

# ntherm = 1/(exp((OmegaS)/(TempMax))-1)
# Fth = (1/(ntherm*(ntherm+1)*(2*ntherm+1)*(2*ntherm+1)))*((exp(OmegaS/TempMax)*OmegaS)/((-1+exp(OmegaS/TempMax))*(-1+exp(OmegaS/TempMax))*TempMax*TempMax))

###############################################################################
###################################  Hamiltonian ##############################

HS = (OmegaS/2) * Sz # Sistema 
Hs = (OmegaS/2) * sz # Sistema sozinho

HA1 = (Omega1/2) * Az1 # Ancilla 1
HA2 = (Omega2/2) * Az2 # Ancilla 2
HA3 = (Omega3/2) * Az3 # Ancilla 3
HA4 = (Omega4/2) * Az4 # Ancilla 4
HA5 = (Omega5/2) * Az5 # Ancilla 5
HA6 = (Omega6/2) * Az6 # Ancilla 6
HA7 = (Omega7/2) * Az7 # Ancilla 7
HA8 = (Omega8/2) * Az8 # Ancilla 8
 
VSA1 = g*(Sp*Am1 + Sm*Ap1) # Troca sistema + Ancilla 1
VSA2 = g*(Sp*Am2 + Sm*Ap2) # Troca sistema + Ancilla 2
VSA3 = g*(Sp*Am3 + Sm*Ap3) # Troca sistema + Ancilla 3
VSA4 = g*(Sp*Am4 + Sm*Ap4) # Troca sistema + Ancilla 4
VSA5 = g*(Sp*Am5 + Sm*Ap5) # Troca sistema + Ancilla 5
VSA6 = g*(Sp*Am6 + Sm*Ap6) # Troca sistema + Ancilla 6
VSA7 = g*(Sp*Am7 + Sm*Ap7) # Troca sistema + Ancilla 7
VSA8 = g*(Sp*Am8 + Sm*Ap8) # Troca sistema + Ancilla 8

VA1A2 = g*(Ap1*Am2 + Am1*Ap2) # Troca Ancila 1 + Ancilla 2
VA2A3 = g*(Ap2*Am3 + Am2*Ap3) # Troca Ancila 2 + Ancilla 3
VA3A4 = g*(Ap3*Am4 + Am3*Ap4) # Troca Ancila 3 + Ancilla 4
VA4A5 = g*(Ap4*Am5 + Am4*Ap5) # Troca Ancila 4 + Ancilla 5
VA5A6 = g*(Ap5*Am6 + Am5*Ap6) # Troca Ancila 5 + Ancilla 6
VA6A7 = g*(Ap6*Am7 + Am6*Ap7) # Troca Ancila 6 + Ancilla 7
VA7A8 = g*(Ap7*Am8 + Am7*Ap8) # Troca Ancila 7 + Ancilla 8

HSA1 = HS + HA1 + VSA1 # Sistema + Ancilla 1
HSA2 = HS + HA2 + VSA2 # Sistema + Ancilla 2
HSA3 = HS + HA3 + VSA3 # Sistema + Ancilla 3
HSA4 = HS + HA4 + VSA4 # Sistema + Ancilla 4
HSA5 = HS + HA5 + VSA5 # Sistema + Ancilla 5
HSA6 = HS + HA6 + VSA6 # Sistema + Ancilla 6
HSA7 = HS + HA7 + VSA7 # Sistema + Ancilla 7
HSA8 = HS + HA8 + VSA8 # Sistema + Ancilla 8

HA1A2 = HA1 + HA2 + VA1A2 # Ancilla 1 + Ancilla 2
HA2A3 = HA2 + HA3 + VA2A3 # Ancilla 2 + Ancilla 3
HA3A4 = HA3 + HA4 + VA3A4 # Ancilla 3 + Ancilla 4
HA4A5 = HA4 + HA5 + VA4A5 # Ancilla 4 + Ancilla 5
HA5A6 = HA5 + HA6 + VA5A6 # Ancilla 5 + Ancilla 6
HA6A7 = HA6 + HA7 + VA6A7 # Ancilla 6 + Ancilla 7
HA7A8 = HA7 + HA8 + VA7A8 # Ancilla 7 + Ancilla 8
###############################################################################
r = -1
# q = 0
for T in Temp:
    # print(T)
    
    # q = q + 1
    r = r + 1
    print(r)
###############################################################################
###############################  Thermal Number  ##############################
    # nt = 1/(exp((hp*Omega)/(kb*T))-1)
    nt = 1/(exp((OmegaS)/(T))-1)
    nthermo[r] = nt
    # print(nt)
    # sech = 1/cosh(x)
    FthGab[r] = ((OmegaS/(T**2))**2)*(1/(np.cosh(OmegaS/T)))**2
###############################################################################
###############################  Collapse Operator  ###########################

    C1S = np.sqrt(gamma*(nt+1))*sm
    C2S = np.sqrt(gamma*nt)*sp

    C1 = np.sqrt(gamma*(nt+1))*Sm
    C2 = np.sqrt(gamma*nt)*Sp

    Clist = [C1,C2]

    ClistS = [C1S,C2S]

###############################################################################
#################################  Initial State  #############################

    G = basis(2,0) # base: excited state
    E = basis(2,1) # base: ground state


#######  Sistema
    S0 = G
    # S0 = 1/(sqrt(2))*(G+E)
    # S0mat = S0*S0.dag()

#######  Ancilla 1
    A1_st = G
    # A1_st = 1/(sqrt(2))*(G+E)
    A1_mat = A1_st*A1_st.dag()
    # print(A_mat)
#######  Ancilla 2
    A2_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A2_mat = A2_st*A2_st.dag()
    # # print(A_mat)
#######  Ancilla 3
    A3_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A3_mat = A3_st*A3_st.dag()
    # # print(A_mat)
 #######  Ancilla 4   
    A4_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A4_mat = A4_st*A4_st.dag()
    # # print(A_mat)
 #######  Ancilla 5   
    A5_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A5_mat = A5_st*A5_st.dag()
    # # print(A_mat)
 #######  Ancilla 6  
    A6_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A6_mat = A6_st*A6_st.dag()
    # # print(A_mat)
 #######  Ancilla 7
    A7_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A7_mat = A7_st*A7_st.dag()
    # # print(A_mat)
 #######  Ancilla 8
    A8_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A8_mat = A8_st*A8_st.dag()
    # # print(A_mat)
    
    psi = tensor(S0,A1_st,A2_st,A3_st,A4_st,A5_st,A6_st,A7_st,A8_st)
    # rho0 = psi*psi.dag()
#######  Completo
    # psi = tensor(S_st[-1],A1_st,A2_st)
    S0mat = psi*psi.dag()

    medataSE = mesolve(HS,S0mat,tSE,[Clist],[]) # Master equation evolution - Sistema + Ambiente
    rho0 = medataSE.states # Take matrices in each time
    rho0 = rho0[-1] # Take matrices in each time
    # A_evo = rho0.ptrace([1])
    # print(A_evo)

###############################################################################

###############################################################################
########################  Master equation solution  ###########################
    s = -1
    for x in range(0, n):
        s = s + 1
        # print(x)
        # S_evo = rho0.ptrace([0]) # Estado da Sistema
        # print(S_evo)
        # A_evo = rho0.ptrace([1]) # Estado da Ancilla
        # print(A_evo)

        ######### Sistema + Ancilla 1
        # rhoSA1 = rho0[-1].ptrace([0,1])
        medataSA1 = mesolve(HSA1,rho0,tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 1
        exptSA1 = medataSA1.states # Take matrices in each time
    
        Rho_S1 = exptSA1[-1].ptrace([0]) # Estado do Sistema
        Rho_A1U1 = exptSA1[-1].ptrace([1]) #Estado da Ancilla 1
        # print(Rho_A1U1)
        
        #######################################################################
        ######### Ancilla 1 + Ancilla 2
        # rhoA1A2 = tensor(Rho_A1U1,A2_mat)
        medataA1A2 = mesolve(HA1A2,exptSA1[-1],tA1A2,[],[]) # Master equation evolution - Ancila 1 + Ancilla 2
        exptA1A2 = medataA1A2.states # Take matrices in each time
        
        Rho_A1U2 = exptA1A2[-1].ptrace([1]) # Estado do Ancila 1
        Rho_A2U1 = exptA1A2[-1].ptrace([2]) #Estado da Ancilla 2
        
        ###############################
        ######### Sistema + Ancilla 2
        medataSA2 = mesolve(HSA2,exptA1A2[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 2
        exptSA2 = medataSA2.states # Take matrices in each time
        
        Rho_S2 = exptSA2[-1].ptrace([0]) # Estado do Sistema
        Rho_A2U2 = exptSA2[-1].ptrace([2]) #Estado da Ancilla 2
        # Rho_A1ciclo = exptSA2[-1].ptrace([1]) # Estado do Ancila 1

        #######################################################################
        ######### Ancilla 2 + Ancilla 3
        # rhoA1A2 = tensor(Rho_A1U1,A2_mat)
        medataA2A3 = mesolve(HA2A3,exptSA2[-1],tA2A3,[],[]) # Master equation evolution - Ancila 2 + Ancilla 3
        exptA2A3 = medataA2A3.states # Take matrices in each time
        
        Rho_A2U3 = exptA2A3[-1].ptrace([2]) # Estado do Ancila 2
        Rho_A3U1 = exptA2A3[-1].ptrace([3]) #Estado da Ancilla 3
        # print(Rho_A3U1)
        
        ###############################
        ######### Sistema + Ancilla 3
        medataSA3 = mesolve(HSA3,exptA2A3[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 3
        exptSA3 = medataSA3.states # Take matrices in each time
        
        Rho_S3 = exptSA3[-1].ptrace([0]) # Estado do Sistema
        Rho_A3U2 = exptSA3[-1].ptrace([3]) #Estado da Ancilla 3
        # Rho_A1ciclo = exptSA3[-1].ptrace([1]) # Estado do Ancila 1
        # print(Rho_A3U2)
        
        #######################################################################
        ######### Ancilla 3 + Ancilla 4
        medataA3A4 = mesolve(HA3A4,exptSA3[-1],tA3A4,[],[]) # Master equation evolution - Ancila 3 + Ancilla 4
        exptA3A4 = medataA3A4.states # Take matrices in each time
        
        Rho_A3U3 = exptA3A4[-1].ptrace([3]) #Estado da Ancilla 3
        Rho_A4U1 = exptA3A4[-1].ptrace([4]) # Estado do Ancila 4
        # print(Rho_A4U1)
        
        ###############################
        ######### Sistema + Ancilla 4
        medataSA4 = mesolve(HSA4,exptA3A4[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 4
        exptSA4 = medataSA4.states # Take matrices in each time
        
        Rho_S4 = exptSA4[-1].ptrace([0]) # Estado do Sistema
        Rho_A4U2 = exptSA4[-1].ptrace([4]) #Estado da Ancilla 4
        # Rho_A1ciclo = exptSA4[-1].ptrace([1]) # Estado do Ancila 1
        
        #######################################################################
        ######### Ancilla 4 + Ancilla 5
        medataA4A5 = mesolve(HA4A5,exptSA4[-1],tA4A5,[],[]) # Master equation evolution - Ancila 4 + Ancilla 5
        exptA4A5 = medataA4A5.states # Take matrices in each time
        
        Rho_A4U3 = exptA4A5[-1].ptrace([4]) #Estado da Ancilla 4
        Rho_A5U1 = exptA4A5[-1].ptrace([5]) # Estado do Ancila 5
        # print(Rho_A4U1)
        
        ###############################
        ######### Sistema + Ancilla 5
        medataSA5 = mesolve(HSA5,exptA4A5[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 5
        exptSA5 = medataSA5.states # Take matrices in each time
        
        Rho_S5 = exptSA5[-1].ptrace([0]) # Estado do Sistema
        Rho_A5U2 = exptSA5[-1].ptrace([5]) #Estado da Ancilla 5
        # Rho_A1ciclo = exptSA4[-1].ptrace([1]) # Estado do Ancila 1
        
        #######################################################################
        ######### Ancilla 5 + Ancilla 6
        medataA5A6 = mesolve(HA5A6,exptSA5[-1],tA5A6,[],[]) # Master equation evolution - Ancila 5 + Ancilla 6
        exptA5A6 = medataA5A6.states # Take matrices in each time
        
        Rho_A5U3 = exptA5A6[-1].ptrace([5]) #Estado da Ancilla 5
        Rho_A6U1 = exptA5A6[-1].ptrace([6]) # Estado do Ancila 6
        # print(Rho_A4U1)
        
        ###############################
        ######### Sistema + Ancilla 6
        medataSA6 = mesolve(HSA6,exptA5A6[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 6
        exptSA6 = medataSA6.states # Take matrices in each time
        
        Rho_S6 = exptSA6[-1].ptrace([0]) # Estado do Sistema
        Rho_A6U2 = exptSA6[-1].ptrace([6]) #Estado da Ancilla 6
        # Rho_A1ciclo = exptSA6[-1].ptrace([1]) # Estado do Ancila 1
        
        #######################################################################
        ######### Ancilla 6 + Ancilla 7
        medataA6A7 = mesolve(HA6A7,exptSA6[-1],tA6A7,[],[]) # Master equation evolution - Ancila 6 + Ancilla 7
        exptA6A7 = medataA6A7.states # Take matrices in each time
        
        Rho_A6U3 = exptA6A7[-1].ptrace([6]) #Estado da Ancilla 6
        Rho_A7U1 = exptA6A7[-1].ptrace([7]) # Estado do Ancila 7
        # print(Rho_A4U1)
        
        ###############################
        ######### Sistema + Ancilla 7
        medataSA7 = mesolve(HSA7,exptA6A7[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 7
        exptSA7 = medataSA7.states # Take matrices in each time
        
        Rho_S7 = exptSA7[-1].ptrace([0]) # Estado do Sistema
        Rho_A7U2 = exptSA7[-1].ptrace([7]) #Estado da Ancilla 7
        # Rho_A1ciclo = exptSA7[-1].ptrace([1]) # Estado do Ancila 1
        
        #######################################################################
        ######### Ancilla 7 + Ancilla 8
        medataA7A8 = mesolve(HA7A8,exptSA7[-1],tA7A8,[],[]) # Master equation evolution - Ancila 7 + Ancilla 8
        exptA7A8 = medataA7A8.states # Take matrices in each time
        
        Rho_A7U3 = exptA7A8[-1].ptrace([7]) #Estado da Ancilla 7
        Rho_A8U1 = exptA7A8[-1].ptrace([8]) # Estado do Ancila 8
        # print(Rho_A4U1)
        
        ###############################
        ######### Sistema + Ancilla 8
        medataSA8 = mesolve(HSA8,exptA7A8[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 8
        exptSA8 = medataSA8.states # Take matrices in each time
        
        Rho_S8 = exptSA8[-1].ptrace([0]) # Estado do Sistema
        Rho_A8U2 = exptSA8[-1].ptrace([8]) #Estado da Ancilla 8
        Rho_A1ciclo = exptSA8[-1].ptrace([1]) # Estado do Ancila 1
        
        #######################################################################
        ######### Sistema + Environment
    # medataSE = mesolve(HS,exptSA[-1],tSE,[Clist],[]) # Master equation evolution - Sistema + Ambiente
        medataSE = mesolve(Hs,Rho_S8,tSE,[ClistS],[]) # Master equation evolution - Sistema
        exptSE = medataSE.states # Take matrices in each time
        
        rho0 = tensor(exptSE[-1],A1_mat,A2_mat,A3_mat,A4_mat,A5_mat,A6_mat,A7_mat,A8_mat)
        
###############################################################################    
###############################   POVM Ancilla 1   ############################  
        q = -1
        for i in range(len(theta)):
            q = q + 1
            z = -1
            for p in range(len(phi)):
                z = z + 1
                # mat1 = Qobj([[exp(-1j*phi[p])], [exp(1j*phi[p])*cos(theta[i])]])
                mat1 = Qobj([[cos(theta[i])], [exp(1j*phi[p])*sin(theta[i])]])
                P1a = mat1*mat1.dag()
            # P1 = tensor(qeye(2),P1a)
            
            # P2 = -1j*exp(1j*phi[p])*sin(theta[i])*sm
                mat2 = Qobj([[exp(-1j*phi[p])*sin(theta[i])],[-cos(theta[i])]])
                P2a = mat2*mat2.dag()
            # P2 = tensor(qeye(2),P2a)

                p1A1 = (P1a*Rho_Rho_S1).tr()
                # p1A1 = (P1a*Rho_A1U2).tr()
                # p1A1 = (P1a*Rho_A1ciclo).tr()
                # print(p1)
                p2A1 = (P2a*Rho_Rho_S1).tr()
                # p2A1 = (P2a*Rho_A1U2).tr()
                # p2A1 = (P2a*Rho_A1ciclo).tr()
            
#             # p1 = (P1*exptSA[-1]).tr()
#             # # print(p1)
#             # p2 = (P2*exptSA[-1]).tr()
            
                p1dA1[r,s,q,z] = real(p1A1)
                p2dA1[r,s,q,z] = real(p2A1)
            
                p1lnA1 = np.log(p1A1)
                p2lnA1 = np.log(p2A1)
                # print(p1ln)
                
                p1lndA1[r,s,q,z] = p1lnA1
                p2lndA1[r,s,q,z] = p2lnA1

###############################################################################                
###############################   POVM Ancilla 2   ############################  
        q = -1
        for i in range(len(theta)):
            q = q + 1
            z = -1
            for p in range(len(phi)):
                z = z + 1
                # mat1 = Qobj([[exp(-1j*phi[p])], [exp(1j*phi[p])*cos(theta[i])]])
                mat1 = Qobj([[cos(theta[i])], [exp(1j*phi[p])*sin(theta[i])]])
                P1a = mat1*mat1.dag()
            # P1 = tensor(qeye(2),P1a)
            
            # P2 = -1j*exp(1j*phi[p])*sin(theta[i])*sm
                mat2 = Qobj([[exp(-1j*phi[p])*sin(theta[i])],[-cos(theta[i])]])
                P2a = mat2*mat2.dag()
            # P2 = tensor(qeye(2),P2a)

                p1A2 = (P1a*Rho_S2).tr()
                # print(p1)
                p2A2 = (P2a*Rho_S2).tr()
            
#             # p1 = (P1*exptSA[-1]).tr()
#             # # print(p1)
#             # p2 = (P2*exptSA[-1]).tr()
            
                p1dA2[r,s,q,z] = real(p1A2)
                p2dA2[r,s,q,z] = real(p2A2)
            
                p1lnA2 = np.log(p1A2)
                p2lnA2 = np.log(p2A2)
                # print(p1ln)
                
                p1lndA2[r,s,q,z] = p1lnA2
                p2lndA2[r,s,q,z] = p2lnA2
                
###############################################################################                
###############################   POVM Ancilla 3   ############################  
        q = -1
        for i in range(len(theta)):
            q = q + 1
            z = -1
            for p in range(len(phi)):
                z = z + 1
                # mat1 = Qobj([[exp(-1j*phi[p])], [exp(1j*phi[p])*cos(theta[i])]])
                mat1 = Qobj([[cos(theta[i])], [exp(1j*phi[p])*sin(theta[i])]])
                P1a = mat1*mat1.dag()
            # P1 = tensor(qeye(2),P1a)
            
            # P2 = -1j*exp(1j*phi[p])*sin(theta[i])*sm
                mat2 = Qobj([[exp(-1j*phi[p])*sin(theta[i])],[-cos(theta[i])]])
                P2a = mat2*mat2.dag()
            # P2 = tensor(qeye(2),P2a)

                p1A3 = (P1a*Rho_S3).tr()
                # print(p1)
                p2A3 = (P2a*Rho_S3).tr()
            
#             # p1 = (P1*exptSA[-1]).tr()
#             # # print(p1)
#             # p2 = (P2*exptSA[-1]).tr()
            
                p1dA3[r,s,q,z] = real(p1A3)
                p2dA3[r,s,q,z] = real(p2A3)
            
                p1lnA3 = np.log(p1A3)
                p2lnA3 = np.log(p2A3)
                # print(p1ln)
                
                p1lndA3[r,s,q,z] = p1lnA3
                p2lndA3[r,s,q,z] = p2lnA3
                
###############################################################################                
###############################   POVM Ancilla 4   ############################  
        q = -1
        for i in range(len(theta)):
            q = q + 1
            z = -1
            for p in range(len(phi)):
                z = z + 1
                # mat1 = Qobj([[exp(-1j*phi[p])], [exp(1j*phi[p])*cos(theta[i])]])
                mat1 = Qobj([[cos(theta[i])], [exp(1j*phi[p])*sin(theta[i])]])
                P1a = mat1*mat1.dag()
            # P1 = tensor(qeye(2),P1a)
            
            # P2 = -1j*exp(1j*phi[p])*sin(theta[i])*sm
                mat2 = Qobj([[exp(-1j*phi[p])*sin(theta[i])],[-cos(theta[i])]])
                P2a = mat2*mat2.dag()
            
                p1A4 = (P1a*Rho_S4).tr()
                # print(p1)
                p2A4 = (P2a*Rho_S4).tr()
                        
                p1dA4[r,s,q,z] = real(p1A4)
                p2dA4[r,s,q,z] = real(p2A4)
            
                p1lnA4 = np.log(p1A4)
                p2lnA4 = np.log(p2A4)
                # print(p1ln)
                
                p1lndA4[r,s,q,z] = p1lnA4
                p2lndA4[r,s,q,z] = p2lnA4
                   
###############################################################################                
###############################   POVM Ancilla 5   ############################  
        q = -1
        for i in range(len(theta)):
            q = q + 1
            z = -1
            for p in range(len(phi)):
                z = z + 1
                # mat1 = Qobj([[exp(-1j*phi[p])], [exp(1j*phi[p])*cos(theta[i])]])
                mat1 = Qobj([[cos(theta[i])], [exp(1j*phi[p])*sin(theta[i])]])
                P1a = mat1*mat1.dag()
            
                mat2 = Qobj([[exp(-1j*phi[p])*sin(theta[i])],[-cos(theta[i])]])
                P2a = mat2*mat2.dag()

                p1A5 = (P1a*Rho_S5).tr()
                # print(p1)
                p2A5 = (P2a*Rho_S5).tr()
            
                p1dA5[r,s,q,z] = real(p1A5)
                p2dA5[r,s,q,z] = real(p2A5)
            
                p1lnA5 = np.log(p1A5)
                p2lnA5 = np.log(p2A5)
                # print(p1ln)
                
                p1lndA5[r,s,q,z] = p1lnA5
                p2lndA5[r,s,q,z] = p2lnA5
                
###############################################################################                
###############################   POVM Ancilla 6   ############################  
        q = -1
        for i in range(len(theta)):
            q = q + 1
            z = -1
            for p in range(len(phi)):
                z = z + 1
                # mat1 = Qobj([[exp(-1j*phi[p])], [exp(1j*phi[p])*cos(theta[i])]])
                mat1 = Qobj([[cos(theta[i])], [exp(1j*phi[p])*sin(theta[i])]])
                P1a = mat1*mat1.dag()
            
                mat2 = Qobj([[exp(-1j*phi[p])*sin(theta[i])],[-cos(theta[i])]])
                P2a = mat2*mat2.dag()

                p1A6 = (P1a*Rho_S6).tr()
                # print(p1)
                p2A6 = (P2a*Rho_S6).tr()
            
                p1dA6[r,s,q,z] = real(p1A6)
                p2dA6[r,s,q,z] = real(p2A6)
            
                p1lnA6 = np.log(p1A6)
                p2lnA6 = np.log(p2A6)
                # print(p1ln)
                
                p1lndA6[r,s,q,z] = p1lnA6
                p2lndA6[r,s,q,z] = p2lnA6

###############################################################################                
###############################   POVM Ancilla 7   ############################  
        q = -1
        for i in range(len(theta)):
            q = q + 1
            z = -1
            for p in range(len(phi)):
                z = z + 1
                # mat1 = Qobj([[exp(-1j*phi[p])], [exp(1j*phi[p])*cos(theta[i])]])
                mat1 = Qobj([[cos(theta[i])], [exp(1j*phi[p])*sin(theta[i])]])
                P1a = mat1*mat1.dag()
            
                mat2 = Qobj([[exp(-1j*phi[p])*sin(theta[i])],[-cos(theta[i])]])
                P2a = mat2*mat2.dag()

                p1A7 = (P1a*Rho_S7).tr()
                # print(p1)
                p2A7 = (P2a*Rho_S7).tr()
            
                p1dA7[r,s,q,z] = real(p1A7)
                p2dA7[r,s,q,z] = real(p2A7)
            
                p1lnA7 = np.log(p1A7)
                p2lnA7 = np.log(p2A7)
                # print(p1ln)
                
                p1lndA7[r,s,q,z] = p1lnA7
                p2lndA7[r,s,q,z] = p2lnA7
                
###############################################################################                
###############################   POVM Ancilla 8   ############################  
        q = -1
        for i in range(len(theta)):
            q = q + 1
            z = -1
            for p in range(len(phi)):
                z = z + 1
                # mat1 = Qobj([[exp(-1j*phi[p])], [exp(1j*phi[p])*cos(theta[i])]])
                mat1 = Qobj([[cos(theta[i])], [exp(1j*phi[p])*sin(theta[i])]])
                P1a = mat1*mat1.dag()
            
                mat2 = Qobj([[exp(-1j*phi[p])*sin(theta[i])],[-cos(theta[i])]])
                P2a = mat2*mat2.dag()

                p1A8 = (P1a*Rho_S8).tr()
                # print(p1)
                p2A8 = (P2a*Rho_S8).tr()
            
                p1dA8[r,s,q,z] = real(p1A8)
                p2dA8[r,s,q,z] = real(p2A8)
            
                p1lnA8 = np.log(p1A8)
                p2lnA8 = np.log(p2A8)
                # print(p1ln)
                
                p1lndA8[r,s,q,z] = p1lnA8
                p2lndA8[r,s,q,z] = p2lnA8
                
###############################################################################
##################################   Ancilla 1  ###############################
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                derP1A1[r,x,i,p] = (p1lndA1[r+1,x,i,p] - p1lndA1[r,x,i,p])/dTemp
                derP2A1[r,x,i,p] = (p2lndA1[r+1,x,i,p] - p2lndA1[r,x,i,p])/dTemp
    
    
for r in range(len(Temp)-1): 
# print(r)
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                f1 = p1dA1[r,x,i,p]*(derP1A1[r,x,i,p])**2
                f2 = p2dA1[r,x,i,p]*(derP2A1[r,x,i,p])**2
                    
                    # print(f1)
                fx = f1+f2
            
                    # FI = max(fx)
    
                F1A1[r,x,i,p] = f1
                F2A1[r,x,i,p] = f2
    
                FxA1[r,x,i,p] = fx
    
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
            QFI_tA1[r,x,i] = max(FxA1[r,x,i,:])


for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA1[r,x] = max(QFI_tA1[r,x,:])
            

# plot(NN,QFIA1[-1,:])
# plot(ddTemp,QFIA1[:,-1])

###############################################################################
##################################   Ancilla 2  ###############################
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                derP1A2[r,x,i,p] = (p1lndA2[r+1,x,i,p] - p1lndA2[r,x,i,p])/dTemp
                derP2A2[r,x,i,p] = (p2lndA2[r+1,x,i,p] - p2lndA2[r,x,i,p])/dTemp
    
    
for r in range(len(Temp)-1): 
# print(r)
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                f1 = p1dA2[r,x,i,p]*(derP1A2[r,x,i,p])**2
                f2 = p2dA2[r,x,i,p]*(derP2A2[r,x,i,p])**2
                    
                    # print(f1)
                fx = f1+f2
            
                    # FI = max(fx)
    
                F1A2[r,x,i,p] = f1
                F2A2[r,x,i,p] = f2
    
                FxA2[r,x,i,p] = fx
    
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
            QFI_tA2[r,x,i] = max(FxA2[r,x,i,:])
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA2[r,x] = max(QFI_tA2[r,x,:])
            
        
###############################################################################
##################################   Ancilla 3  ###############################
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                derP1A3[r,x,i,p] = (p1lndA3[r+1,x,i,p] - p1lndA3[r,x,i,p])/dTemp
                derP2A3[r,x,i,p] = (p2lndA3[r+1,x,i,p] - p2lndA3[r,x,i,p])/dTemp
    
    
for r in range(len(Temp)-1): 
# print(r)
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                f1 = p1dA3[r,x,i,p]*(derP1A3[r,x,i,p])**2
                f2 = p2dA3[r,x,i,p]*(derP2A3[r,x,i,p])**2
                    
                    # print(f1)
                fx = f1+f2
            
                    # FI = max(fx)
    
                F1A3[r,x,i,p] = f1
                F2A3[r,x,i,p] = f2
    
                FxA3[r,x,i,p] = fx
    
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
            QFI_tA3[r,x,i] = max(FxA3[r,x,i,:])

for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA3[r,x] = max(QFI_tA3[r,x,:])

# plot(NN,QFIA2[-1,:],'--r')
# plot(ddTemp,QFIA2[:,-1],'--r')


###############################################################################
##################################   Ancilla 4  ###############################
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                derP1A4[r,x,i,p] = (p1lndA4[r+1,x,i,p] - p1lndA4[r,x,i,p])/dTemp
                derP2A4[r,x,i,p] = (p2lndA4[r+1,x,i,p] - p2lndA4[r,x,i,p])/dTemp
    
    
for r in range(len(Temp)-1): 
# print(r)
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                f1 = p1dA4[r,x,i,p]*(derP1A4[r,x,i,p])**2
                f2 = p2dA4[r,x,i,p]*(derP2A4[r,x,i,p])**2
                    
                    # print(f1)
                fx = f1+f2
            
                    # FI = max(fx)
    
                F1A4[r,x,i,p] = f1
                F2A4[r,x,i,p] = f2
    
                FxA4[r,x,i,p] = fx
    
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
            QFI_tA4[r,x,i] = max(FxA4[r,x,i,:])

for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA4[r,x] = max(QFI_tA4[r,x,:])

# plot(NN,QFIA2[-1,:],'--r')
# plot(ddTemp,QFIA2[:,-1],'--r')

###############################################################################
##################################   Ancilla 5  ###############################
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                derP1A5[r,x,i,p] = (p1lndA5[r+1,x,i,p] - p1lndA5[r,x,i,p])/dTemp
                derP2A5[r,x,i,p] = (p2lndA5[r+1,x,i,p] - p2lndA5[r,x,i,p])/dTemp
    
    
for r in range(len(Temp)-1): 
# print(r)
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                f1 = p1dA5[r,x,i,p]*(derP1A5[r,x,i,p])**2
                f2 = p2dA5[r,x,i,p]*(derP2A5[r,x,i,p])**2
                    
                    # print(f1)
                fx = f1+f2
            
                    # FI = max(fx)
    
                F1A5[r,x,i,p] = f1
                F2A5[r,x,i,p] = f2
    
                FxA5[r,x,i,p] = fx
    
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
            QFI_tA5[r,x,i] = max(FxA5[r,x,i,:])

for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA5[r,x] = max(QFI_tA5[r,x,:])

# plot(NN,QFIA2[-1,:],'--r')
# plot(ddTemp,QFIA2[:,-1],'--r')

###############################################################################
##################################   Ancilla 6  ###############################
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                derP1A6[r,x,i,p] = (p1lndA6[r+1,x,i,p] - p1lndA6[r,x,i,p])/dTemp
                derP2A6[r,x,i,p] = (p2lndA6[r+1,x,i,p] - p2lndA6[r,x,i,p])/dTemp
    
    
for r in range(len(Temp)-1): 
# print(r)
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                f1 = p1dA6[r,x,i,p]*(derP1A6[r,x,i,p])**2
                f2 = p2dA6[r,x,i,p]*(derP2A6[r,x,i,p])**2
                    
                    # print(f1)
                fx = f1+f2
            
                    # FI = max(fx)
    
                F1A6[r,x,i,p] = f1
                F2A6[r,x,i,p] = f2
    
                FxA6[r,x,i,p] = fx
    
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
            QFI_tA6[r,x,i] = max(FxA6[r,x,i,:])

for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA6[r,x] = max(QFI_tA6[r,x,:])

# plot(NN,QFIA2[-1,:],'--r')
# plot(ddTemp,QFIA2[:,-1],'--r')

###############################################################################
##################################   Ancilla 7  ###############################
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                derP1A7[r,x,i,p] = (p1lndA7[r+1,x,i,p] - p1lndA7[r,x,i,p])/dTemp
                derP2A7[r,x,i,p] = (p2lndA7[r+1,x,i,p] - p2lndA7[r,x,i,p])/dTemp
    
    
for r in range(len(Temp)-1): 
# print(r)
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                f1 = p1dA7[r,x,i,p]*(derP1A7[r,x,i,p])**2
                f2 = p2dA7[r,x,i,p]*(derP2A7[r,x,i,p])**2
                    
                    # print(f1)
                fx = f1+f2
            
                    # FI = max(fx)
    
                F1A7[r,x,i,p] = f1
                F2A7[r,x,i,p] = f2
    
                FxA7[r,x,i,p] = fx
    
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
            QFI_tA7[r,x,i] = max(FxA7[r,x,i,:])

for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA7[r,x] = max(QFI_tA7[r,x,:])

# plot(NN,QFIA2[-1,:],'--r')
# plot(ddTemp,QFIA2[:,-1],'--r')

###############################################################################
##################################   Ancilla 8  ###############################
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                derP1A8[r,x,i,p] = (p1lndA8[r+1,x,i,p] - p1lndA8[r,x,i,p])/dTemp
                derP2A8[r,x,i,p] = (p2lndA8[r+1,x,i,p] - p2lndA8[r,x,i,p])/dTemp
    
    
for r in range(len(Temp)-1): 
# print(r)
    for x in range(0, n):
        for i in range(len(theta)):
                # print(i)
            for p in range(len(phi)):
                f1 = p1dA8[r,x,i,p]*(derP1A8[r,x,i,p])**2
                f2 = p2dA8[r,x,i,p]*(derP2A8[r,x,i,p])**2
                    
                    # print(f1)
                fx = f1+f2
            
                    # FI = max(fx)
    
                F1A8[r,x,i,p] = f1
                F2A8[r,x,i,p] = f2
    
                FxA8[r,x,i,p] = fx
    
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        for i in range(len(theta)):
            QFI_tA8[r,x,i] = max(FxA8[r,x,i,:])

for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA8[r,x] = max(QFI_tA8[r,x,:])

# plot(NN,QFIA2[-1,:],'--r')
# plot(ddTemp,QFIA2[:,-1],'--r')

###############################################################################
# FthSca = np.zeros((len(Temp)-1))
# for r in range(len(Temp)-1):
    
#     FthSca[r] = (1/(nthermo[r+1]*(nthermo[r+1]+1)*(2*nthermo[r+1]+1)**2))*((nthermo[r+1]-nthermo[r])/dTemp)**2

###############################################################################
plot(ddTemp,QFIA1[:,-1],ddTemp,QFIA2[:,-1],'--r',ddTemp,QFIA3[:,-1],'-.k',ddTemp,QFIA4[:,-1],':m',ddTemp,QFIA5[:,-1],ddTemp,QFIA6[:,-1],'--g',ddTemp,QFIA7[:,-1],'-.y',ddTemp,QFIA8[:,-1],':b')

# plot(NN,QFIA1[12,:],NN,QFIA2[12,:],'--r',NN,QFIA3[12,:],'-.k',NN,QFIA4[12,:],':m',NN,QFIA5[12,:],NN,QFIA6[12,:],'--g',NN,QFIA7[12,:],'-.y',NN,QFIA8[12,:],':b')
# plot(NN,QFIA1[12,:],NN,QFIA2[12,:],'--r',NN,QFIA3[12,:],'-.k',NN,QFIA4[12,:],':m',NN,QFIA5[12,:],NN,QFIA6[13,:],'--g',NN,QFIA7[13,:],'-.y',NN,QFIA8[13,:],':b')

nCam = 8
ACam = linspace(1,nCam,nCam)  # Time S-A

# MaxQFI1 = max(QFIA1[:,-1])
# MaxQFI2 = max(QFIA2[:,-1])
# MaxQFI3 = max(QFIA3[:,-1])
# MaxQFI4 = max(QFIA4[:,-1])
# MaxQFI5 = max(QFIA5[:,-1])

MaxQFI1 = max(QFIA1[12,:])
MaxQFI2 = max(QFIA2[12,:])
MaxQFI3 = max(QFIA3[12,:])
MaxQFI4 = max(QFIA4[12,:])
MaxQFI5 = max(QFIA5[12,:])
MaxQFI6 = max(QFIA6[13,:])
MaxQFI7 = max(QFIA7[13,:])
MaxQFI8 = max(QFIA8[13,:])

QFImax = np.zeros(len(ACam))

QFImax[0] = MaxQFI1
QFImax[1] = MaxQFI2
QFImax[2] = MaxQFI3
QFImax[3] = MaxQFI4
QFImax[4] = MaxQFI5
QFImax[5] = MaxQFI6
QFImax[6] = MaxQFI7
QFImax[7] = MaxQFI8

# plot(ACam,QFImax)

# plot(ddTemp,QFIA1[:,-1]/FthSca,ddTemp,QFIA2[:,-1]/FthSca,'--r')

# plot(NN,QFIA1[13,:]/FthSca[13],NN,QFIA2[13,:]/FthSca[13],'--r')
# plot(NN,QFIA1[13,:]/(FthGab[13]/n),NN,QFIA2[13,:]/(FthGab[13]/n),'--r')
# plot(NN,QFIA1[13,:]/FthGab[13],NN,QFIA2[13,:]/FthGab[13],'--r')
# plot(NN,QFIA1[13,:]/(FthSca[13]/n),NN,QFIA2[13,:]/(FthSca[13]/n),'--r')

# plt.xscale("log")
# plt.yscale("log")
# plt.plot(NN,QFIA1[13,:]/(FthGab[13]/n),NN,QFIA2[13,:]/(FthGab[13]/n),'--r')
# plt.plot(NN,QFIA1[12,:]/FthGab[12],NN,QFIA2[12,:]/FthGab[12],'--r')
# plt.plot(NN,QFIA1[13,:]/FthGab[13],NN,QFIA2[13,:]/FthGab[13],'--r')

# plt.plot(NN,QFIA1[13,:]/FthSca[13],NN,QFIA2[13,:]/FthSca[13],'--r')

output_data = np.vstack((ddTemp))
file_data_store('ddTemp_t2.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((NN))
# file_data_store('n_Ancila_t2.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((ACam))
# file_data_store('n_camadas_t2.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFImax))
# file_data_store('QFImax_t2.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# ######################################


output_data = np.vstack((QFIA1[:,-1]))
file_data_store('QFI_SA1_varT_t100.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

output_data = np.vstack((QFIA2[:,-1]))
file_data_store('QFIA2_SA2_varT_t100.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

output_data = np.vstack((QFIA3[:,-1]))
file_data_store('QFIA3_SA3_varT_t100.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

output_data = np.vstack((QFIA4[:,-1]))
file_data_store('QFIA4_SA4_varT_t100.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

output_data = np.vstack((QFIA5[:,-1]))
file_data_store('QFIA5_SA5_varT_t100.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

output_data = np.vstack((QFIA6[:,-1]))
file_data_store('QFIA6_SA6_varT_t100.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

output_data = np.vstack((QFIA7[:,-1]))
file_data_store('QFIA7_SA7_varT_t100.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

output_data = np.vstack((QFIA8[:,-1]))
file_data_store('QFIA8_SA8_varT_t100.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")


#######################################
# output_data = np.vstack((QFIA1[12,:]))
# file_data_store('QFIA1_Nanc_T12.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA1[13,:]))
# file_data_store('QFIA1_Nanc_T13.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA2[12,:]))
# file_data_store('QFIA2_Nanc_T12.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA2[13,:]))
# file_data_store('QFIA2_Nanc_T13.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA3[12,:]))
# file_data_store('QFIA3_Nanc_T12.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA3[13,:]))
# file_data_store('QFIA3_Nanc_T13.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA4[12,:]))
# file_data_store('QFIA4_Nanc_T12.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA4[13,:]))
# file_data_store('QFIA4_Nanc_T13.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA5[12,:]))
# file_data_store('QFIA5_Nanc_T12.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA5[13,:]))
# file_data_store('QFIA5_Nanc_T13.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA6[12,:]))
# file_data_store('QFIA6_Nanc_T12.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA6[13,:]))
# file_data_store('QFIA6_Nanc_T13.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA7[12,:]))
# file_data_store('QFIA7_Nanc_T12.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA7[13,:]))
# file_data_store('QFIA7_Nanc_T13.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA8[12,:]))
# file_data_store('QFIA8_Nanc_T12.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((QFIA8[13,:]))
# file_data_store('QFIA8_Nanc_T13.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")
###############################################################################
###############################################################################


tend = time.time()    # tempo final de processamento
delta = tend - tin    # funcao calculo do intervalo de tempo  de processamento
print (delta)


