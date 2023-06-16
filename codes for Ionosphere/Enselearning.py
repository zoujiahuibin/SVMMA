from sklearn import svm
import numpy as np


def addone(X):
    return np.hstack((np.ones((len(X),1)),X))

class Bagging():
    def __init__(self,X_train,Y_train,Models,Indexs):
        self.w=None
        self.X_train=X_train
        self.Y_train = Y_train
        self.Indexs=Indexs
        self.Models=Models


    def get_weights(self,type='unif'):

        if type=='unif':
            self.w=np.ones(len(self.Models))/len(self.Models)
        elif type=='optimal':
            A = np.zeros((len(self.Y_train), 1))
            for clf, index in zip(self.Models, self.Indexs):
                X_train_s = self.X_train[:, index]
                A = np.hstack((A, clf.predict(X_train_s).reshape(-1, 1)))
            A = A[:, 1:]
            A = np.mat(A)
            self.w=np.array(np.linalg.pinv(A.T*A)*A.T*self.Y_train.reshape((-1,1)))


    def predict(self,X_Pre):
        Pre=np.zeros(X_Pre.shape[0])
        for clf,w,index in zip(self.Models,self.w,self.Indexs):
            X_Pre_s = X_Pre[:, index]
            Pre+=w*clf.predict(X_Pre_s)
        self.Pre=np.sign(Pre)
        return self.Pre

class Adaboosting():
    def __init__(self,X_train,Y_train,Indexs,C=1):
        self.w=None
        self.X_train=X_train
        self.Y_train = Y_train
        self.Indexs=Indexs
        self.Models=[]
        self.Indexs2=[]
        self.order=[]
        self.C=C

    def fit(self,Iter_max):
        Dt=np.ones(self.X_train.shape[0])/self.X_train.shape[0]
        iter=1
        w2=[]
        # for index in self.Indexs:
        while True:
            ind0 = np.random.choice(range(self.Indexs.shape[0]), )
            index = self.Indexs[ind0, :]

            clf = svm.LinearSVC(penalty='l2', loss='hinge', C=self.C)
            X_train_s = self.X_train[:, index]

            ind=np.random.choice(range(X_train_s.shape[0]), size=X_train_s.shape[0], replace=True, p=Dt)

            X_train_s_t=X_train_s[ind,:]
            clf.fit(X_train_s_t, self.Y_train)
            beta = np.hstack((clf.intercept_, clf.coef_[0]))
            Y_res_t = np.sign(np.dot(addone(X_train_s), beta))
            epsilon=1-(Y_res_t==self.Y_train).mean()

            self.Models.append(clf)
            alpha=0.5*np.log(1/epsilon-1)
            w2.append(alpha)
            self.Indexs2.append(index)
            self.order.append(ind0)
            if iter > Iter_max - 1:
                break
            else:
                iter += 1
            Dt=Dt*np.exp(-alpha*Y_res_t*self.Y_train)
            Dt=Dt/sum(Dt)
        weights = np.zeros(self.Indexs.shape[0])
        for w, ind in zip(w2, self.order):
            weights[ind] += w
        self.w=weights



    def predict(self,X_Pre):
        Pre=np.zeros(X_Pre.shape[0])
        for clf,w,index in zip(self.Models,self.w,self.Indexs2):
            X_Pre_s = X_Pre[:, index]
            Pre+=w*clf.predict(X_Pre_s)
        self.Pre=np.sign(Pre)
        return self.Pre


