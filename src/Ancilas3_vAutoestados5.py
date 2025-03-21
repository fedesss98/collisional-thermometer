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


##########  QFI com 3 camadas de Ancillas


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
# tSA1 = linspace(0.00001,pi/100,tp)  # Time S-A
tA1A2 = linspace(0.00001,pi/2,tp)  # Time A-A
tA2A3 = linspace(0.00001,pi/2,tp)  # Time A-A
tSE = linspace(0.00001,0.1,tp)  # Time S-E
# tSE = linspace(0.00001,0.1,tp)  # Time S-E

td = linspace(0.0,5.0,tp-1)
# dt = 5/tp
dtSA1 = tSA1[1]-tSA1[0]
dtA1A2 = tA1A2[1]-tA1A2[0]

n = 30
NN = range(0, n)
Nn = linspace(1.0,30,n)
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

# derRhoS1 = np.zeros((len(Temp)-1,len(range(0, n)),len(tSA1)))
# derRhoA1 = np.zeros((len(Temp)-1,len(range(0, n)),len(tSA1)))

derRhoS1 = np.zeros((len(tSA1)))
derRhoA1 = np.zeros((len(tSA1)))

# Rho_S1 = np.zeros((len(Temp),len(range(0, n))))
Rho_S1l11 = np.zeros((len(Temp),len(range(0, n))))
Rho_S1l12 = np.zeros((len(Temp),len(range(0, n))))
Rho_S1l21 = np.zeros((len(Temp),len(range(0, n))))
Rho_S1l22 = np.zeros((len(Temp),len(range(0, n))))
# Rho_A1U1 = np.zeros((len(Temp),len(range(0, n))))

# Rho_S1 = np.zeros((len(Temp),len(range(0, n))))
Rho_A11l11 = np.zeros((len(Temp),len(range(0, n))))
Rho_A11l12 = np.zeros((len(Temp),len(range(0, n))))
Rho_A11l21 = np.zeros((len(Temp),len(range(0, n))))
Rho_A11l22 = np.zeros((len(Temp),len(range(0, n))))

dRho_A11l11 = np.zeros((len(Temp),len(range(0, n))))
dRho_A11l12 = np.zeros((len(Temp),len(range(0, n))))
dRho_A11l21 = np.zeros((len(Temp),len(range(0, n))))
dRho_A11l22 = np.zeros((len(Temp),len(range(0, n))))

H1 = np.zeros((len(Temp)-1))
H2 = np.zeros((len(Temp)-1))
H3 = np.zeros((len(Temp)-1))
H4 = np.zeros((len(Temp)-1))
H5 = np.zeros((len(Temp)-1))
H6 = np.zeros((len(Temp)-1))
H7 = np.zeros((len(Temp)-1))
H8 = np.zeros((len(Temp)-1))
H9 = np.zeros((len(Temp)-1))
H10 = np.zeros((len(Temp)-1))

HA11 = np.zeros((len(Temp),len(range(0, n))))
HA12 = np.zeros((len(Temp)))
HA21 = np.zeros((len(Temp)))
HA22 = np.zeros((len(Temp)))

A11val_Lamb1 = (len(Temp),len(range(0, n)))
A11val_Lamb2 = (len(Temp),len(range(0, n)))

A11vet1_Lamb1 = (len(Temp),len(range(0, n)))
A11vet2_Lamb1 = (len(Temp),len(range(0, n)))
A11vet1_Lamb2 = (len(Temp),len(range(0, n)))
A11vet2_Lamb2 = (len(Temp),len(range(0, n)))

##########################

p1dA1 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2dA1 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

p1lndA1 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))
p2lndA1 = np.zeros((len(Temp),len(range(0, n)),len(theta),len(phi)))

derP1A1 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))
derP2A1 = np.zeros((len(Temp)-1,len(range(0, n)),len(theta),len(phi)))

# #################
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

PA = np.zeros(len(NN))
###############################################################################
#################################  Operadores  ################################

N=2

# System Alone:
sm= sigmap()
sp= sigmam()
sx= sigmax()
sy= sigmay()
sz= sigmaz()

# System + 3 Ancillas:
# System
Sm = tensor(sigmap(),qeye(2),qeye(2),qeye(2))
Sp = tensor(sigmam(),qeye(2),qeye(2),qeye(2))
Sx = tensor(sigmax(),qeye(2),qeye(2),qeye(2))
Sy = tensor(sigmay(),qeye(2),qeye(2),qeye(2))
Sz = tensor(sigmaz(),qeye(2),qeye(2),qeye(2))

# print(Sp.ptrace(0))

# Ancilla 1:
Am1 = tensor(qeye(2),sigmap(),qeye(2),qeye(2))
Ap1 = tensor(qeye(2),sigmam(),qeye(2),qeye(2))
Ax1 = tensor(qeye(2),sigmax(),qeye(2),qeye(2))
Ay1 = tensor(qeye(2),sigmay(),qeye(2),qeye(2))
Az1 = tensor(qeye(2),sigmaz(),qeye(2),qeye(2))

# Ancilla 2:
Am2 = tensor(qeye(2),qeye(2),sigmap(),qeye(2))
Ap2 = tensor(qeye(2),qeye(2),sigmam(),qeye(2))
Ax2 = tensor(qeye(2),qeye(2),sigmax(),qeye(2))
Ay2 = tensor(qeye(2),qeye(2),sigmay(),qeye(2))
Az2 = tensor(qeye(2),qeye(2),sigmaz(),qeye(2))

# Ancilla 3:
Am3 = tensor(qeye(2),qeye(2),qeye(2),sigmap())
Ap3 = tensor(qeye(2),qeye(2),qeye(2),sigmam())
Ax3 = tensor(qeye(2),qeye(2),qeye(2),sigmax())
Ay3 = tensor(qeye(2),qeye(2),qeye(2),sigmay())
Az3 = tensor(qeye(2),qeye(2),qeye(2),sigmaz())
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
 
VSA1 = g*(Sp*Am1 + Sm*Ap1) # Troca sistema + Ancilla 1
VSA2 = g*(Sp*Am2 + Sm*Ap2) # Troca sistema + Ancilla 2
VSA3 = g*(Sp*Am3 + Sm*Ap3) # Troca sistema + Ancilla 3

VA1A2 = g*(Ap1*Am2 + Am1*Ap2) # Troca Ancilla 1 + Ancilla 2
VA2A3 = g*(Ap2*Am3 + Am2*Ap3) # Troca Ancila 2 + Ancilla 3

HSA1 = HS + HA1 + VSA1 # Sistema + Ancilla 1
HSA2 = HS + HA2 + VSA2 # Sistema + Ancilla 2
HSA3 = HS + HA3 + VSA3 # Sistema + Ancilla 3

HA1A2 = HA1 + HA2 + VA1A2 # Ancilla 1 +Ancilla 2
HA2A3 = HA2 + HA3 + VA2A3 # Ancilla 1 +Ancilla 2
###############################################################################
r = -1
# q = 0
G = basis(2,0) # base: excited state
A1_st = G
# A1_st = 1/(sqrt(2))*(G+E)
A1_mat = A1_st*A1_st.dag()
RA01 = A1_mat
RA012 = A1_mat
RA02 = A1_mat
RA022 = A1_mat
RA03 = A1_mat
RA032 = A1_mat


A10_evalue, A10_evector = LA.eig(RA01)
A10vet_Lamb1 = Qobj(A10_evector[0])
A10vet_Lamb2 = Qobj(A10_evector[1])
A10val_Lamb1 = A10_evalue[0]
A10val_Lamb2 = A10_evalue[1]



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
    # FthGab[r] = ((OmegaS/(T**2))**2)*(1/(np.cosh(OmegaS/T)))**2
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

#######  Ancilla
    A1_st = G
    # A1_st = 1/(sqrt(2))*(G+E)
    A1_mat = A1_st*A1_st.dag()
    # print(A_mat)
    Lamb1 = A1_mat[0,0]
    Lamb2 = A1_mat[1,1]
    RA01 = A1_mat

    A2_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A2_mat = A2_st*A2_st.dag()
    # # print(A_mat)

    A3_st = G
    # A2_st = 1/(sqrt(2))*(G+E)
    A3_mat = A3_st*A3_st.dag()
    # # print(A_mat)
    
    psi = tensor(S0,A1_st,A2_st,A3_st)
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
        medataSA1 = mesolve(HSA1,rho0,tSA1,[],[]) # Master equation evolution - Sistema + Ancilla
        exptSA1 = medataSA1.states # Take matrices in each time
    
        Rho_S1 = exptSA1[-1].ptrace([0]) # Estado do Sistema
        # Rho_S1l11[r,s] = Rho_S1[0,0]
        # Rho_S1l12[r,s] = Rho_S1[0,1]
        # Rho_S1l21[r,s] = Rho_S1[1,0]
        # Rho_S1l22[r,s] = Rho_S1[1,1]
        
        Rho_A1U1 = exptSA1[-1].ptrace([1]) #Estado da Ancilla 1
        Rho_A11l11[r,s] = Rho_A1U1[0,0]
        Rho_A11l12[r,s] = Rho_A1U1[0,1]
        Rho_A11l21[r,s] = Rho_A1U1[1,0]
        Rho_A11l22[r,s] = Rho_A1U1[1,1]
        # print(Rho_A1U1)
        
        ######### Sistema + Environment
    # medataSE = mesolve(HS,exptSA[-1],tSE,[Clist],[]) # Master equation evolution - Sistema + Ambiente
        medataSE = mesolve(Hs,Rho_S1,tSE,[ClistS],[]) # Master equation evolution - Sistema
        exptSE = medataSE.states # Take matrices in each time
        
        rho0 = tensor(exptSE[-1],A1_mat,A2_mat,A3_mat)
        # rho0 = exptSA1[-1]
        
##############################################################################          
        

        DerRA1U1 = Rho_A1U1/dTemp - RA01/dTemp
        RA01 = Rho_A1U1
        dRho_A11l11 = Rho_A1U1[0,0]
        dRho_A11l12 = Rho_A1U1[0,1]
        dRho_A11l21 = Rho_A1U1[1,0]
        dRho_A11l22 = Rho_A1U1[1,1]
        # print(dRho_A11l11)
        # print(dRho_A11l12)
        # print(dRho_A11l21)
        # print(dRho_A11l22)
        
        HA11[r,s] =  ((dRho_A11l11)*(dRho_A11l11))/(Lamb1 + Rho_A1U1[0,0])+((dRho_A11l22)*(dRho_A11l22))/(Lamb2 +Rho_A1U1[1,1])
        # print(HA1_tot)
        Lamb1 = Rho_A1U1[0,0]# Autovalor do estado anterior
        Lamb2 = Rho_A1U1[1,1]# Autovalor do estado anterior
        
        # dRho_A11l11[r,s] = Rho_A1U1[0,0]
        # dRho_A11l12[r,s] = Rho_A1U1[0,1]
        # dRho_A11l21[r,s] = Rho_A1U1[1,0]
        # dRho_A11l22[r,s] = Rho_A1U1[1,1]
        
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

                p1A1 = (P1a*Rho_A1U1).tr()
                # p1A1 = (P1a*Rho_A1U2).tr()
                # p1A1 = (P1a*Rho_A1ciclo).tr()
                # print(p1)
                p2A1 = (P2a*Rho_A1U1).tr()
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

###################################################################################
    
        
    # DerRA1U1 = Rho_A1U1/dTemp - RA01/dTemp
    # RA01 = Rho_A1U1
    # dRho_A11l11 = Rho_A1U1[0,0]
    # dRho_A11l12 = Rho_A1U1[0,1]
    # dRho_A11l21 = Rho_A1U1[1,0]
    # dRho_A11l22 = Rho_A1U1[1,1]
    # print(dRho_A11l11)
    # print(dRho_A11l12)
    # print(dRho_A11l21)
    # print(dRho_A11l22)
    
    # # HA11 =  ((dRho_A11l11)*(dRho_A11l11))/(2*Rho_A1U1[0,0])+((dRho_A11l22)*(dRho_A11l22))/(2*Rho_A1U1[1,1])
    
    # HA11 =  ((dRho_A11l11)*(dRho_A11l11))/(Lamb1 + Rho_A1U1[0,0])+((dRho_A11l22)*(dRho_A11l22))/(Lamb2 +Rho_A1U1[1,1])
    # # print(HA1_tot)
    # Lamb1 = Rho_A1U1[0,0]# Autovalor do estado anterior
    # Lamb2 = Rho_A1U1[1,1]# Autovalor do estado anterior
    
    # print(HA11)
    
    

                   
# # ###############################################################################
# ##################################   Ancilla 1  ###############################
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


# # # for r in range(len(Temp)-1):
# # #     for x in range(0, n):
# # #         for i in range(len(phi)):
# # #             QFI_t[r,x,i] = max(Fx[r,x,:,i])
            
# # # for r in range(len(Temp)-1):
# # #     for x in range(0, n):
# # #         QFI_t[r,x] = max(Fx[r,x,:,:])
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA1[r,x] = max(QFI_tA1[r,x,:])
            

# plot(NN,QFIA1[-1,:])
# plot(ddTemp,QFIA1[:,-1])
plot(Temp,HA11[:,-1])



# output_data = np.vstack((Rho_A11l11[:,-1]))
# file_data_store('Rho_A11l11_T13n30.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((Rho_A11l12[:,-1]))
# file_data_store('Rho_A11l12_T13n30.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((Rho_A11l21[:,-1]))
# file_data_store('Rho_A11l21_T13n30.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((Rho_A11l22[:,-1]))
# file_data_store('Rho_A11l22_T13n30.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")


# output_data = np.vstack((Rho_A11l11[13,:]))
# file_data_store('Rho_A11l11_T13nVar.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((Rho_A11l12[13,:]))
# file_data_store('Rho_A11l12_T13nVar.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((Rho_A11l21[13,:]))
# file_data_store('Rho_A11l21_T13nVar.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((Rho_A11l22[13,:]))
# file_data_store('Rho_A11l22_T13nVar.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")


# output_data = np.vstack((ddTemp[:]))
# file_data_store('TempDeriva.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((Temp[:]))
# file_data_store('Temp.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")

# output_data = np.vstack((NN[:]))
# file_data_store('nAncilla.dat', output_data.T, numtype="real", numformat="decimal", sep= " ")


###############################################################################################

tend = time.time()    # tempo final de processamento
delta = tend - tin    # funcao calculo do intervalo de tempo  de processamento
print (delta)


