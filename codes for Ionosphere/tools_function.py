import numpy as np
from sklearn import svm
import select_method
import scipy.stats as stat


def SIC(bic):
    # 该函数用于计算sbic
    sbic = np.array([1 / sum(np.exp((bic[s] - bic) / 2)) for s in range(len(bic))])
    ind = np.where(sbic == np.inf)
    sbic[ind] = max(sbic != np.inf) + 1e3
    return sbic

def addone(X):
    return np.hstack((np.ones((len(X),1)),X))

def Indexs_fun(p, Mn):
    a = range(p)  # 全排变量的位次
    Mn = min(Mn, p)
    indexs = []
    for i in range(1, Mn):
        indexs += list(itertools.combinations(a, i))
    Indexs = np.repeat(0, len(indexs) * p).reshape(len(indexs), p)
    for i in range(len(indexs)):
        Indexs[i, indexs[i]] = 1
    # Indexs[:, 0:m] = 1
    return Indexs == 1



def Indexs_fun_kmeans(X,k):
    #通过聚类feature，进行模型索引的准备
    KMeans=myKMeans(X=X,k=k,tol=1e-4,maxiter=50)
    C,labels=KMeans.fit(axis=1,normtype=None)

    varclass=[]
    for i in range(k):
        varclass.append(np.where(labels==i)[0].tolist())

    #从每一类中挑选出一个变量
    Indexs=np.zeros((1,X.shape[1]))
    while sum(map(len,varclass)):
        index=np.zeros((1,X.shape[1]))
        for i in range(k):
            if varclass[i]:
                index[0,varclass[i].pop()]=1
        Indexs=np.vstack((Indexs,index))
    Indexs=Indexs[1:,:]

    return Indexs==1





def Indexs_fun_lasso(X,Y,k=None):
    # choose the first k features, and choose all the x defaultly
    #sort the varibales by lasso, then nest to build up the first model set
    if not k:
        k=X.shape[1]
    features_ind=[]
    for lam in np.linspace(10,0.001,max(50,X.shape[1]//5)):
        clf=svm.LinearSVC(C=lam,penalty='l1',dual=False,max_iter=10000)
        # 注意C越小，penalty的压力越大
        clf.fit(X,Y)
        beta=clf.coef_[0]
        beta_ind=np.where(beta==0)[0]
        new=list(set(beta_ind)-set(features_ind))
        features_ind.extend(new)
    new=list(set(range(X.shape[1]))-set(features_ind))
    features_ind.extend(new)
    features_ind=features_ind[::-1]
    features_ind=features_ind[0:k]

    #build the candidate models by nest
    Indexs=np.ones(X.shape[1])
    for i in range(1,len(features_ind)+1):
        ones=np.zeros(X.shape[1])
        ones[features_ind[0:i]]=1
        Indexs=np.vstack((Indexs,ones))
    Indexs=Indexs[1:,:]
    return Indexs==1




def Indexs_fun_lasso_select(X,Y,C,k=None):
    #k: the amount of candidate models
    # step1：lasso to generate first model set
    # step2: using cl, ch1, ch2, ch4 to choose k submodels to build up the final model set
    ## step1:
    if not k:
        k=X.shape[1]
    Indexs=Indexs_fun_lasso(X, Y, X.shape[1])
    ## step2:
    Models = []
    Betas=[]
    HingesLoss=[]
    for index in Indexs:
        clf=svm.LinearSVC(penalty='l2',loss='hinge', C=C)
        X_train_s = X[:, index]
        clf.fit(X_train_s, Y)
        beta = np.hstack((clf.intercept_, clf.coef_[0]))
        del clf
        Betas.append(beta)
        tmp = map(lambda x: max(x, 0), 1 - Y * np.dot(addone(X_train_s), beta))  # hinge loss
        HingesLoss.append(sum(tmp))



    CH_method=select_method.SVMICH(HingesLoss,Indexs,Models,X.shape[0],Ln=np.sqrt(np.log(X.shape[0])))
    CH_method.get_scores()
    # model_inds=CL_method.SVMICL_scores
    model_inds=sorted(range(len(CH_method.SVMICH_scores)), key=lambda k: CH_method.SVMICH_scores[k]) [0:min(k,X.shape[1])] # 返回排序索引,从小到大排列
    Indexs=Indexs[model_inds,:]
    # Indexs[0:4,:]=True
    return Indexs



