
from copy import deepcopy
import numpy as np
import pandas as pd

#实现了将相关性强的聚为了一类
class myKMeans():
    # Euclidean Distance Caculator
    def __init__(self,X,k,tol=1e-4,maxiter=500):
        self.X=X# X: np.narray,the data
        self.k=k# k:int<len(X),Number of clusters
        self.tol=tol#float, if the distance between two centroids is smaller tol, the computation will stop
        self.maxiter=maxiter#int, the maximal circulation turns


    def dist(self,x,ax=None):
        rho=abs(np.corrcoef(x[0],x[1])[1,0])
        return 1e6 if np.isnan(rho) else 1/rho-1

    def dist2(self,x, ax=None):
        #used for matrix the distance between centroids
        #if you want any other distance, please modify this part
        return np.linalg.norm(x[0] - x[1], axis=ax)

    def fit(self,axis=0,normtype=None):
        # X: np.narray,the data
        #axis: 0 represents cluster for samples, and 1 for features
        # k:int<len(X),Number of clusters
        #normtype: is the same as ax in np.linalg.norm
        #return: C is the center of data, clusters is the label

        if axis:
            X=self.X.T
        else:
            X=self.X
        np.random.seed(100)
        C_ind=np.random.choice(range(len(X)),size=self.k,replace=False)
        C=X[C_ind,:]# C is the center axis and a row is a point


        # To store the value of centroids when it updates
        C_old = np.zeros(C.shape)
        # Cluster Lables(0, 1, 2)
        clusters = np.zeros(len(X))
        # Error func. - Distance between new centroids and old centroids
        error = max(map(lambda x: self.dist2(x, normtype), zip(C, C_old)))
        # Loop will run till the error becomes zero
        circum=0
        while error >self.tol:
            if circum>=self.maxiter:
                break
            # Assigning each value to its closest cluster
            for i in range(len(X)):
                distances =map(lambda x:self.dist(x,normtype),((X[i],C[j]) for j in range(self.k)))
                cluster = np.argmin(list(distances))
                clusters[i] = cluster
            # Storing the old centroid values
            C_old = deepcopy(C)
            # Finding the new centroids by taking the average value
            for i in range(self.k):
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                C[i] = np.mean(points, axis=0)
            error = self.dist2((C, C_old), None)
            circum+=1

        return C,clusters

if __name__ == '__main__':
    import os

    os.chdir('C:\\Users\\Administrator\\Desktop\\friendly-fortnight-master\\')
    data = pd.read_csv('xclara.csv')
    X=np.array(list(zip(data['V1'],data['V2'])))
    X=X.T
    KMeans=myKMeans(X=X,k=3,tol=1e-4,maxiter=10)
    C,labels=KMeans.fit(axis=1,normtype=None)
