import numpy as np
import matplotlib.pyplot as plt
import MajdiPerceptron as MPerc
import MajdiUtils as MUtil


# mData = MUtil.GetData('toy_data.tsv','\t')
# labels = mData[:,0]
# vectors = mData[:,1:]

# mData = MUtil.GetData('pima-indians-diabetes.txt',',')
# labels = mData[:,-1]
# labels[labels==0]=-1
# vectors = mData[:,:-1]

#GET DATA
center1 = np.array([0,0])
center2 = np.array([4,3])
nbsamples = 400
variance = 0.7
bluepoints = np.random.multivariate_normal(mean=center2,cov=np.identity(2) * variance,size=nbsamples)
fbp= np.c_[bluepoints,np.ones(nbsamples)]
redpoints = np.random.multivariate_normal(mean=center1,cov=np.identity(2) * variance,size=nbsamples)
frp = np.c_[redpoints,-np.ones(nbsamples)]
rData = np.concatenate((fbp, frp), axis=0)
mData = rData[np.random.permutation(2*nbsamples)]
labels = mData[:,-1]
vectors = mData[:,:-1]

#DISPLAY DATA
colors = ['b' if label == 1 else 'r' for label in labels]
plt.scatter(vectors[:,0], vectors[:,1], s=40, c=colors)
xmin, xmax = plt.axis()[:2]
plt.show()


#RUN ALGORITHMS
(nRows, nCols)= mData.shape
myRange = nRows

cT0_S = cT0_A = cT0_O = cT0_P =0
cT_S = cT_A = cT_O = cT_P = np.zeros(nCols-1)
n=0

mExecData = np.zeros((1,4*(nCols+1)))


mExecData[0,-4]=mExecData[0,-3]=mExecData[0,-2]=mExecData[0,-1]=MUtil.GetErrorRate(cT_S,cT0_S,labels,vectors)


for i in range(myRange):
    n+=1
    label = labels[i]
    vector = vectors[i]
    df= 1/np.sqrt(n)
    nT_S,nT0_S = MPerc.PerceptronStep(cT_S,cT0_S,label,vector)
    nT_A,nT0_A = MPerc.PerceptronStepA(n,cT_A,cT0_A,nT_S,nT0_S)
    nT_P,nT0_P = MPerc.PerceptronStepP(cT_P,cT0_P,label,vector,df)

    TE_S = MUtil.GetErrorRate(nT_S,nT0_S,labels,vectors)
    TE_A = MUtil.GetErrorRate(nT_A,nT0_A,labels,vectors)
    TE_P = MUtil.GetErrorRate(nT_P,nT0_P,labels,vectors)

    nT_O, nT0_O, TE_O = MPerc.PerceptronStepO(nT_S,nT0_S,TE_S,mExecData,nCols)

    newLine = np.zeros((1,4*(nCols+1)))
    for j in range(nCols-1):
        newLine[0,j]=nT_S[j]
        newLine[0,j+nCols]=nT_A[j]
        newLine[0,(j+2*nCols)]= nT_O[j]
        newLine[0,(j+2*nCols)]= nT_P[j]

    newLine[0,nCols-1]=nT0_S
    newLine[0,2*nCols-1]=nT0_A
    newLine[0,3*nCols-1]=nT0_O
    newLine[0,4*nCols-1]=nT0_P

    newLine[0,4*nCols]=TE_S
    newLine[0,4*nCols+1]=TE_A
    newLine[0,4*nCols+2]=TE_O
    newLine[0,4*nCols+3]=TE_P

    mExecData = np.append(mExecData,newLine,axis=0)
    cT_S = nT_S
    cT0_S = nT0_S
    cT_A = nT_A
    cT0_A = nT0_A
    cT_P = nT_P
    cT0_P = nT0_P

x = np.linspace(0,myRange,myRange+1)
y_S = mExecData[:,4*nCols]
y_A = mExecData[:,4*nCols+1]
y_O = mExecData[:,4*nCols+2]
y_P = mExecData[:,4*nCols+3]

#PERCEPTRON IS RED
plt.plot(x,y_S,color='red')

#AVERAGED PERCEPTRON IS BLUE
plt.plot(x,y_A,color='blue')

#CUSTOMIZED PERCEPTRON IS ORANGE
plt.plot(x,y_O,color='orange')

#PEGASOS IS GREEN
plt.plot(x,y_P,color='green')

plt.style.use('dark_background')
plt.show()

# plt.xlabel('Nb of iterations')
# plt.ylabel('Errors')
# plt.title('Classification Error Rate')
# plt.show()

print("OK")

