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

#######  Ancilla
    A1_st = G
    # A1_st = 1/(sqrt(2))*(G+E)
    A1_mat = A1_st*A1_st.dag()
    # print(A_mat)

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
        Rho_A1U1 = exptSA1[-1].ptrace([1]) #Estado da Ancilla 1
        # print(Rho_A1U1)
        #######################################################################
        ######### Ancilla 1 + Ancilla 2
        # rhoA1A2 = tensor(Rho_A1U1,A2_mat)
        medataA1A2 = mesolve(HA1A2,exptSA1[-1],tA1A2,[],[]) # Master equation evolution - Sistema + Ancilla
        exptA1A2 = medataA1A2.states # Take matrices in each time
        
        Rho_A1U2 = exptA1A2[-1].ptrace([1]) # Estado do Ancila 1
        Rho_A2U1 = exptA1A2[-1].ptrace([2]) #Estado da Ancilla 2
        #############################
        ######### Sistema + Ancilla 2
        # rho0SA2 = tensor(Rho_S1,qeye(2),Rho_A2U1)
        # medataSA2 = mesolve(HSA2,rho0SA2,tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 2
        medataSA2 = mesolve(HSA2,exptA1A2[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla
        exptSA2 = medataSA2.states # Take matrices in each time
        
        Rho_S2 = exptSA2[-1].ptrace([0]) # Estado do Sistema
        Rho_A2U2 = exptSA2[-1].ptrace([2]) #Estado da Ancilla 2
        # Rho_A1ciclo = exptSA2[-1].ptrace([1]) # Estado do Ancila 1

        #######################################################################
        ######### Ancilla 2 + Ancilla 3
        # rhoA1A2 = tensor(Rho_A1U1,A2_mat)
        medataA2A3 = mesolve(HA2A3,exptSA2[-1],tA2A3,[],[]) # Master equation evolution - Sistema + Ancilla
        exptA2A3 = medataA2A3.states # Take matrices in each time
        
        Rho_A2U3 = exptA2A3[-1].ptrace([2]) # Estado do Ancila 1
        Rho_A3U1 = exptA2A3[-1].ptrace([3]) #Estado da Ancilla 2
        # print(Rho_A3U1)
        #############################
        ######### Sistema + Ancilla 3
        # rho0SA2 = tensor(Rho_S1,qeye(2),Rho_A2U1)
        # medataSA2 = mesolve(HSA2,rho0SA2,tSA1,[],[]) # Master equation evolution - Sistema + Ancilla 2
        medataSA3 = mesolve(HSA3,exptA2A3[-1],tSA1,[],[]) # Master equation evolution - Sistema + Ancilla
        exptSA3 = medataSA3.states # Take matrices in each time
        
        Rho_S3 = exptSA3[-1].ptrace([0]) # Estado do Sistema
        Rho_A3U2 = exptSA3[-1].ptrace([3]) #Estado da Ancilla 2
        Rho_A1ciclo = exptSA2[-1].ptrace([1]) # Estado do Ancila 1
        # print(Rho_A3U2)
        
        #######################################################################
        ######### Sistema + Environment
    # medataSE = mesolve(HS,exptSA[-1],tSE,[Clist],[]) # Master equation evolution - Sistema + Ambiente
        medataSE = mesolve(Hs,Rho_S3,tSE,[ClistS],[]) # Master equation evolution - Sistema
        exptSE = medataSE.states # Take matrices in each time
        
        rho0 = tensor(exptSE[-1],A1_mat,A2_mat,A3_mat)
        
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

                p1A2 = (P1a*Rho_A2U2).tr()
                # print(p1)
                p2A2 = (P2a*Rho_A2U2).tr()
            
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

                p1A3 = (P1a*Rho_A3U2).tr()
                # print(p1)
                p2A3 = (P2a*Rho_A3U2).tr()
            
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

# # for r in range(len(Temp)-1):
# #     for x in range(0, n):
# #         for i in range(len(phi)):
# #             QFI_t[r,x,i] = max(Fx[r,x,:,i])
            
# # for r in range(len(Temp)-1):
# #     for x in range(0, n):
# #         QFI_t[r,x] = max(Fx[r,x,:,:])
    
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

# # for r in range(len(Temp)-1):
# #     for x in range(0, n):
# #         for i in range(len(phi)):
# #             QFI_t[r,x,i] = max(Fx[r,x,:,i])
            
# # for r in range(len(Temp)-1):
# #     for x in range(0, n):
# #         QFI_t[r,x] = max(Fx[r,x,:,:])
    
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

# # for r in range(len(Temp)-1):
# #     for x in range(0, n):
# #         for i in range(len(phi)):
# #             QFI_t[r,x,i] = max(Fx[r,x,:,i])
            
# # for r in range(len(Temp)-1):
# #     for x in range(0, n):
# #         QFI_t[r,x] = max(Fx[r,x,:,:])
    
for r in range(len(Temp)-1):
    for x in range(0, n):
        QFIA3[r,x] = max(QFI_tA3[r,x,:])

# plot(NN,QFIA2[-1,:],'--r')
# plot(ddTemp,QFIA2[:,-1],'--r')

###############################################################################
FthSca = np.zeros((len(Temp)-1))
for r in range(len(Temp)-1):
    
    FthSca[r] = (1/(nthermo[r+1]*(nthermo[r+1]+1)*(2*nthermo[r+1]+1)**2))*((nthermo[r+1]-nthermo[r])/dTemp)**2

plot(ddTemp,QFIA1[:,-1],ddTemp,QFIA2[:,-1],'--r',ddTemp,QFIA3[:,-1],'-.k')
# plot(ddTemp,100*QFIA3[:,-1],'-.k',ddTemp,FthSca)

# plot(NN,QFIA1[13,:],NN,QFIA2[13,:],'--r',NN,QFIA3[13,:],'-.k')

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

###############################################################################
###############################################################################


tend = time.time()    # tempo final de processamento
delta = tend - tin    # funcao calculo do intervalo de tempo  de processamento
print (delta)


