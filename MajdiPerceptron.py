import numpy as np
#cT0: current Theta0 (bias)
#cT: current Theta vector
#label: Classification of current sample
#x: current vector
def PerceptronStep(cT,cT0, label, x):
    nT0 = cT0
    nT = cT
    assess = np.dot(cT,x) + cT0
    if (assess <= 0):
        nT = cT + label * x
        nT0 = cT0 + label     
    return nT,nT0

def PerceptronStepA(n,pT,pT0,nT,nT0):
    rT = (n * pT + nT)/(n+1)
    rT0 = (n * pT0 + nT0)/(n+1)
    return rT,rT0

def PerceptronStepO(t,t0,err,EData,nc):

    rt=t
    rt0 =t0
    rErr = err

    minE = np.min(EData[:,-1])

    if err > minE:
        minIx = np.argmin(EData[:,-1])
        rt = EData[minIx,2*nc:3*nc-1]
        rt0 = EData[minIx,3*nc-1]
        rErr = minE

    return rt,rt0,rErr

def PerceptronStepP(cT,cT0, label, x, eta):
    nT0 = cT0
    nT = cT
    assess = label * (np.dot(x,cT) + cT0)

    if (assess <= 1):
        nT = cT + eta * label * x
        nT0 = cT0 + eta * label 

    return nT,nT0

