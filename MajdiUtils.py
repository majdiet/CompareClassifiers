import numpy as np
import pandas as pd

def GetErrorRate(t,t0,labels,tData):
    R = np.matmul(tData,t) + t0
    R[R>0]=1
    R[R<=0]=-1
    C = (R!=labels)
    result = sum(C)/len(labels)
    return result


def GetData(file,s):
    df = pd.read_csv(file,sep=s,header=None)
    numpyData = df.to_numpy()
    randData = numpyData[np.random.permutation(len(numpyData))]
    return randData


""" N = len(mData)

ix_perm =np.random.permutation(200)
ix_perm_train = ix_perm[:100]
ix_perm_test= ix_perm[100:]

# ix_perm_train = np.array([33,140,19,173,110,166,83,46,50,6,56,144,186,30,69,38,125,96
# ,49,120,152,99,136,149,60,114,128,112,135,184,133,98,3,77,79,190
# ,0,43,147,84,151,11,124,116,176,179,130,158,182,170,127,39,44,65
# ,159,129,10,94,91,167,76,37,18,123,15,141,191,183,119,71,154,174
# ,34,164,165,2,148,146,108,78,31,48,161,138,22,41,193,181,150,25
# ,80,101,75,199,32,105,197,8,121,142])

# ix_perm_test = np.array([194,29,178,134,156,95,13,100,85,169,195,17,92,132,53,72,7,177
# ,9,162,87,52,64,61,62,109,16,47,126,58,57,137,21,70,103,107
# ,40,35,67,198,143,97,180,59,36,113,139,171,42,88,26,102,163,104
# ,4,106,24,192,5,28,157,89,20,196,118,23,185,73,66,111,168,1
# ,81,155,68,188,54,187,122,172,63,189,131,82,55,90,115,145,117,27
# ,93,153,74,14,160,175,12,51,45,86])

mData_train = mData[ix_perm_train]
mData_test = mData[ix_perm_test]

lbl_train = mData_train[:,0]
X_train = mData_train[:,1:]

lbl_test = mData_test[:,0]
X_test = mData_test[:,1:]



ct=np.zeros(2)
ct0 = 0

ErTrn = np.array([MUtil.GetErrorRate(ct,ct0,lbl_train,X_train)])
ErTst = np.array([MUtil.GetErrorRate(ct,ct0,lbl_test,X_test)])

n=len(X_train)

sIx=0
res1=0
res2=0

for t in range(1):
    for i in range(n):
        sIx+=1
        ct, ct0 = MPerc.PerceptronStep(ct,ct0,lbl_train[i],X_train[i])
        res1=np.array([MUtil.GetErrorRate(ct,ct0,lbl_train,X_train)])
        res2=np.array([MUtil.GetErrorRate(ct,ct0,lbl_train,X_train)])
        ErTrn=np.append(ErTrn,res1)
        ErTst=np.append(ErTst,res2)
        print(i,res1,ct,ct0)


# x = np.linspace(0,sIx+1,sIx+1)
# y = ErTrn
# z = ErTst
# plt.plot(x,y)
# plt.xlabel('Nb of iterations')
# plt.ylabel('Errors')
# plt.title('Classification Error Rate')
# plt.show()

colors = ['b' if label == 1 else 'r' for label in mData_train[:,0]]
plt.scatter(X_train[:,0], X_train[:,1], s=40, c=colors)
xmin, xmax = plt.axis()[:2]

x = np.linspace(xmin,xmax,100)
y = (17/9.5411) - (10.8197/9.5411) * x
plt.plot(x,y,color='green')

plt.show()

#[10.8197  9.5411] -17.0

# fig, ax = plt.subplots(1,2,sharex=True)
# ax[0].plot(x, y)
# ax[1].plot(x,z)
# plt.show()



# colors = ['b' if label == 1 else 'r' for label in mData_test[:,0]]
# plt.scatter(mData_test[:, 1], mData_test[:, 2], s=40, c=colors)
# plt.show() """