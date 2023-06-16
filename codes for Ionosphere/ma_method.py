import numpy as np
from sklearn import svm
import scipy.optimize


def addone(X):
    return np.hstack((np.ones((len(X),1)),X))
def MA_Evaluation(weight, X_pre, Y_pre, Models, Indexs):
    beta_weight = 0
    for clf, w, index in zip(Models, weight, Indexs):
        beta = np.zeros(Indexs.shape[1])
        beta[index] = clf.coef_[0]
        beta=np.hstack((clf.intercept_,beta))
        beta_weight = beta * w + beta_weight

    Y_res = np.sign(np.dot(addone(X_pre), beta_weight))
    hingeloss=sum(map(lambda x: max(x,0), 1-Y_pre*np.dot(addone(X_pre),beta_weight)))/len(Y_pre)

    P=sum(Y_pre==1)
    N=sum(Y_pre==-1)
    TP=sum([x==1 and y==1 for x,y in zip(Y_pre,Y_res)])
    TPR=TP/P
    FP=sum([x==-1 and y==1 for x,y in zip(Y_pre,Y_res)])
    FPR=FP/N
    MSE = np.array((Y_res != Y_pre)).mean()
    return TPR,FPR,MSE,hingeloss


class SVMMA2():
    def __init__(self,X_train,Y_train,Indexs,samplesize,Beta,Jn,C):
        self.Beta=Beta
        self.samplesize=samplesize
        self.Jn=Jn
        self.Indexs=Indexs
        self.X_train=X_train
        self.Y_train=Y_train
        self.C=C


    def calculate_weights(self):
        def pre_optima():
            Mn = self.samplesize // self.Jn
            Betas = []
            block = []
            for j in range(self.Jn):
                block.append([])
                if j == self.Jn - 1:
                    block[j] = [not j * Mn <= i < self.samplesize for i in range(self.samplesize)]
                else:
                    block[j] = [not j * Mn <= i < (j + 1) * Mn for i in range(self.samplesize)]
                X_train_j = self.X_train[block[j], :]
                Y_train_j = self.Y_train[block[j]]
                clf=svm.LinearSVC(penalty='l2', loss='hinge', C=self.C,random_state=0,max_iter=10000)
                Betas.append(np.zeros((self.X_train.shape[1]+1, 1)))
                for index in self.Indexs:
                    X_train_s = X_train_j[:, index]
                    clf.fit(X_train_s, Y_train_j)
                    # Models.append(clf)
                    beta = np.zeros(self.X_train.shape[1])
                    beta[index] = clf.coef_[0]
                    beta=np.hstack((clf.intercept_,beta))
                    Betas[j] = np.hstack((Betas[j], beta.reshape(len(beta), 1)))  # 一列是一个beta
                Betas[j] = Betas[j][:, 1:]
            return Betas, block

        def w_object(w, Betas, block):
            Hingelossj = []
            for j in range(self.Jn):
                beta_j = np.dot(Betas[j], w)
                X_train_j = self.X_train[block[j], :]
                Y_train_j = self.Y_train[block[j]]
                Hingelossj.append(sum(map(lambda x: max(x, 0), 1 - Y_train_j * np.dot(addone(X_train_j), beta_j))))
            return sum(Hingelossj) / (self.Jn * self.samplesize)



        bnds = []
        for i in range(self.Indexs.shape[0]):
            bnds.append((0, 1))
        bnds = tuple(bnds)
        p0 = np.zeros(self.Indexs.shape[0])
        Betas, block=pre_optima()
        res = scipy.optimize.minimize(w_object, p0,args=(Betas,block), method='SLSQP',  bounds=bnds,options={'maxiter':200,'disp':False})#,tol=1e-6, constraints=({'type': 'eq', 'fun': lambda w: sum(w) - 1}))
        return res.x




        

    def do_evaluation(self,X_pre,Y_pre):
        w_hat=self.calculate_weights()
        beta_hat=np.dot(self.Beta,w_hat)
        Y_res = np.sign(np.dot(X_pre, beta_hat))
        results['testerror'] = np.array((Y_res != Y_pre)).mean()

    def get_minloss(self,X_pre,Y_pre):

        def w_object3(w_hat,Beta2):
            beta=np.dot(Beta2,w_hat)
            return sum(map(lambda x: max(x,0),1-Y_pre*np.dot(addone(X_pre),beta)))/len(Y_pre)

        Beta2=np.zeros(self.Indexs.shape[1]+1).reshape(self.Indexs.shape[1]+1,1)
        for index,b in zip(self.Indexs,self.Beta):
            beta = np.zeros(self.Indexs.shape[1])
            beta[index] = b
            Beta2=np.hstack((Beta2,beta.reshape(len(beta),1)))
        Beta2=Beta2[:,1:]


        bnds = []
        for i in range(self.Indexs.shape[0]):
            bnds.append((0, 1))
        bnds = tuple(bnds)
        p0 = np.zeros(self.Indexs.shape[0])
        res = scipy.optimize.minimize(w_object3, p0,args=(Beta2), method='SLSQP', bounds=bnds,options={'maxiter': 200,'disp': False})  # ,tol=1e-6, constraints=({'type': 'eq', 'fun': lambda w: sum(w) - 1}))
        return w_object3(res.x,Beta2)


class SVMMA():
    def __init__(self, X_train, Y_train, Indexs, samplesize, Beta, Jn, C):
        self.Beta = Beta
        self.samplesize = samplesize
        self.Jn = Jn
        self.Indexs = Indexs
        self.X_train = X_train
        self.Y_train = Y_train
        self.C = C
        self.beta_ma_=None
        self.w_hat_=None


    def calculate_weights(self):
        def pre_optima():
            Mn = self.samplesize // self.Jn
            Betas = []
            block = []
            for j in range(self.Jn):
                block.append([])
                if j == self.Jn - 1:
                    block[j] = [not j * Mn <= i < self.samplesize for i in range(self.samplesize)]
                else:
                    block[j] = [not j * Mn <= i < (j + 1) * Mn for i in range(self.samplesize)]
                X_train_j = self.X_train[block[j], :]
                Y_train_j = self.Y_train[block[j]]
                clf = svm.LinearSVC(penalty='l2', loss='hinge', C=self.C,random_state=0,max_iter=1000)
                Betas.append(np.zeros((self.X_train.shape[1]+1, 1)))
                for index in self.Indexs:
                    X_train_s = X_train_j[:, index]
                    clf.fit(X_train_s, Y_train_j)
                    beta = np.zeros(self.X_train.shape[1])
                    beta[index] = clf.coef_[0]
                    beta=np.hstack((clf.intercept_,beta))
                    Betas[j] = np.hstack((Betas[j], beta.reshape(len(beta), 1)))  # 一列是一个beta
                Betas[j] = Betas[j][:, 1:]
            return Betas, block

        def w_object(w, Betas, block):
            Hingelossj = []
            for j in range(self.Jn):
                beta_j = np.dot(Betas[j], w)
                notblockj=[not _ for _ in block[j]]
                X_train_j = self.X_train[notblockj, :]
                Y_train_j = self.Y_train[notblockj]
                Hingelossj.append(sum(map(lambda x: max(x, 0), 1 - Y_train_j * np.dot(addone(X_train_j), beta_j))))
            return sum(Hingelossj) / (self.Jn * self.samplesize)

        bnds = []
        for i in range(self.Indexs.shape[0]):
            bnds.append((0, 1))
        bnds = tuple(bnds)
        p0 = np.zeros(self.Indexs.shape[0])
        Betas, block = pre_optima()
        res = scipy.optimize.minimize(w_object, p0, args=(Betas, block), method='SLSQP', bounds=bnds,constraints={'type':'eq','fun':lambda x:sum(x)-1},
                                      options={'maxiter': 200,'disp': False})  # ,tol=1e-6, constraints=({'type': 'eq', 'fun': lambda w: sum(w) - 1}))
        return res.x

    def do_evaluation(self, X_pre, Y_pre):
        w_hat = self.calculate_weights()
        self.w_hat_=w_hat
        beta_hat = np.dot(self.Beta, w_hat)
        self.beta_ma_=beta_hat
        Y_res = np.sign(np.dot(addone(X_pre), beta_hat))
        results['testerror'] = np.array((Y_res != Y_pre)).mean()

    def ROC_Curve(self,X_pre,Y_pre):
        Y_res=np.ones(X_pre.shape[0])
        for t in np.linspace(0,1,21):
            Y_res[np.dot(X_pre,self.beta_ma_)<t]=-1
            TPR.append()
            FPR.append()
        return TPR,FPR




    def get_minloss(self, X_pre, Y_pre):
        def w_object3(w_hat, Beta2):
            beta = np.dot(Beta2, w_hat)
            return sum(map(lambda x: max(x, 0), 1 - Y_pre * np.dot(addone(X_pre), beta))) / len(Y_pre)

        Beta2 = np.zeros(self.Indexs.shape[1]+1).reshape(self.Indexs.shape[1]+1, 1)
        for index, b in zip(self.Indexs, self.Beta):
            beta = np.zeros(self.Indexs.shape[1])
            beta[index] = b[1:]
            beta=np.hstack((b[0],beta))
            Beta2 = np.hstack((Beta2, beta.reshape(-1, 1)))
        Beta2 = Beta2[:, 1:]

        bnds = []
        for i in range(self.Indexs.shape[0]):
            bnds.append((0, 1))
        bnds = tuple(bnds)
        p0 = np.zeros(self.Indexs.shape[0])
        res = scipy.optimize.minimize(w_object3, p0, args=(Beta2), method='SLSQP', bounds=bnds,constraints={'type':'eq','fun':lambda x:sum(x)-1}, options={'maxiter': 200,
                                                                                                         'disp': False})  # ,tol=1e-6, constraints=({'type': 'eq', 'fun': lambda w: sum(w) - 1}))

        return w_object3(res.x, Beta2)
