import numpy as np
from numpy.linalg import cholesky
from sklearn import svm
from sklearn.cluster import KMeans
import itertools
from scipy.special import comb, perm
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat
# from multiprocessing import Pool
from multiprocessing import Process, Queue
import select_method
import ma_method
import matplotlib.pyplot as plt
import figure_file
from mykmeans import myKMeans
from tools_function import *
# print(os.getcwd())
# os.chdir('.\\codes')
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier

import Enselearning

# #计算排列数
# A=perm(3,2)
# #计算组合数
# C=comb(45,2)
# print(A,C)

#模型应该加入截距项，即X有一列是全1的


def main(repeat, X_train, Y_train, X_pre, Y_pre, Mn, S_true, modeltype,ismisspecification):
    # parameter prepration
    samplesize = X_train.shape[0]
    p = X_train.shape[1]
    lambda_n = samplesize ** (-1)
    C = (samplesize * lambda_n) ** (-1)

    # candidate models setting up
    # Indexs = Indexs_fun(p, Mn)
    # k<Mn
    # k<Mn，k表示每个模型最多提供的候选模型个数
    Indexs = Indexs_fun_lasso(X_train, Y_train,k=100)#max(1, int(2*samplesize ** (1 / 3))))
    # estimate for candidate model
    Models = []
    Betas = []
    HingesLoss = []
    for index in Indexs:
        # clf = svm.SVC(kernel='linear', C=C)
        # clf=svm.SVC(kernel='linear', C=C, random_state=0)
        clf=svm.LinearSVC(penalty='l2',loss='hinge', C=C)
        X_train_s = X_train[:, index]
        clf.fit(X_train_s, Y_train)
        Models.append(clf)
        beta = np.hstack((clf.intercept_,clf.coef_[0]))
        Betas.append(beta)
        tmp = map(lambda x: max(x, 0), 1 - Y_train * np.dot(addone(X_train_s), beta))  # hinge loss
        HingesLoss.append(sum(tmp))

    # model selection and weights
    CL_method = select_method.SVMICL(HingesLoss, Indexs, Models, samplesize)
    CH_method = select_method.SVMICH(HingesLoss, Indexs, Models, samplesize, Ln=np.sqrt(np.log(samplesize)))
    w_CL = np.zeros(Indexs.shape[0])
    w_CH = np.zeros(Indexs.shape[0])
    w_full=np.zeros(Indexs.shape[0])
    CL_bestmodel_ind = CL_method.get_bestmodel()
    w_CL[CL_bestmodel_ind] = 1
    CH_bestmodel_ind = CH_method.get_bestmodel()
    w_CH[CH_bestmodel_ind] = 1
    w_full[-1] = 1
    w_unif=np.ones(Indexs.shape[0])/Indexs.shape[0]



    #model averaging
    # k = max(2, int(  len(X_train) ** (1 / 3)))
    # MAset=set()
    # MAset=MAset.union(CL_method.get_bestmodel(k))
    # MAset=MAset.union( CH_method.get_bestmodel(k))
    # MAset=MAset.union(CH_method2.get_bestmodel(k))
    # MAset=MAset.union( CH_method3.get_bestmodel(k))
    # MAset=MAset.union(CH_method4.get_bestmodel(k))
    # MA_inds=list(MAset)


    MA_inds=range(len(Indexs))
    MA_Indexs=Indexs[MA_inds,:]
    MABetas=[Betas[i] for i in MA_inds]
    MA_method = ma_method.SVMMA(X_train, Y_train, MA_Indexs, samplesize, MABetas, Jn=5, C=C)
    MA_method2 = ma_method.SVMMA2(X_train, Y_train, MA_Indexs, samplesize, MABetas, Jn=5, C=C)
    # SVMMA是求和等一，SVMMA2是求和不等一
    zeros = np.zeros(Indexs.shape[0])
    w_ma_hat = MA_method.calculate_weights()
    zeros[MA_inds]=w_ma_hat
    w_ma_hat=zeros
    w_ma_hat2 = MA_method2.calculate_weights()
    zeros = np.zeros(Indexs.shape[0])
    zeros[MA_inds]=w_ma_hat2
    w_ma_hat2=zeros

    w_SCL=SIC(np.array(CL_method.SVMICL_scores)/X_train.shape[0])
    w_SCH=SIC(np.array(CH_method.SVMICH_scores)/X_train.shape[0])


    print('the weights has been gotten.')

    # 整理权重
    weight_df = pd.DataFrame({'w_CL': w_CL,
                              'w_CH': w_CH,
                              # 'w_CH2': w_CH2,
                              # 'w_CH3': w_CH3,
                              # 'w_CH4': w_CH4,
                              'w_ma': w_ma_hat,
                              'w_ma2': w_ma_hat2,
                              'w_SCL':w_SCL,
                              'w_SCH':w_SCH,
                              },)

    # evaluation
    TPR_full, FPR_full, MSE_full, hingeloss_full = ma_method.MA_Evaluation(w_full, X_pre, Y_pre, Models, Indexs)
    TPR_cl, FPR_cl, MSE_cl, hingeloss_cl = ma_method.MA_Evaluation(w_CL, X_pre, Y_pre, Models, Indexs)
    TPR_ch, FPR_ch, MSE_ch, hingeloss_ch = ma_method.MA_Evaluation(w_CH, X_pre, Y_pre, Models, Indexs)
    # TPR_ch2, FPR_ch2, MSE_ch2, hingeloss_ch2 = ma_method.MA_Evaluation(w_CH2, X_pre, Y_pre, Models, Indexs)
    # TPR_ch3, FPR_ch3, MSE_ch3, hingeloss_ch3 = ma_method.MA_Evaluation(w_CH3, X_pre, Y_pre, Models, Indexs)
    # TPR_ch4, FPR_ch4, MSE_ch4, hingeloss_ch4 = ma_method.MA_Evaluation(w_CH4, X_pre, Y_pre, Models, Indexs)
    TPR_ma, FPR_ma, MSE_ma, hingeloss_ma = ma_method.MA_Evaluation(w_ma_hat, X_pre, Y_pre, Models, Indexs)
    TPR_ma2, FPR_ma2, MSE_ma2, hingeloss_ma2 = ma_method.MA_Evaluation(w_ma_hat2, X_pre, Y_pre, Models, Indexs)
    TPR_unif, FPR_unif, MSE_unif, hingeloss_unif = ma_method.MA_Evaluation(w_unif, X_pre, Y_pre, Models, Indexs)
    TPR_SCL, FPR_SCL, MSE_SCL, hingeloss_SCL = ma_method.MA_Evaluation(w_SCL, X_pre, Y_pre, Models, Indexs)
    TPR_SCH, FPR_SCH, MSE_SCH, hingeloss_SCH = ma_method.MA_Evaluation(w_SCH, X_pre, Y_pre, Models, Indexs)

    # implement bagging
    # base_est = svm.LinearSVC(penalty='l2', loss='hinge', C=1, random_state=0, max_iter=1000)
    # clf_bagging = BaggingClassifier(base_estimator=base_est, n_estimators=50, max_features=1.0, random_state=0).fit(
    #     X_train, Y_train)
    # bagging_predict = clf_bagging.predict(X_pre)
    myBagging=Enselearning.Bagging(X_train,Y_train,Models,Indexs)
    myBagging.get_weights(type='unif')
    bagging_predict=myBagging.predict(X_pre)
    MSE_Bagging = (bagging_predict != Y_pre).mean()
    P = sum(Y_pre == 1)
    N = sum(Y_pre == -1)
    TP = sum([x == 1 and y == 1 for x, y in zip(Y_pre, bagging_predict)])
    TPR_Bagging = TP / P
    FP = sum([x == -1 and y == 1 for x, y in zip(Y_pre, bagging_predict)])
    FPR_Bagging = FP / N

    # implement adaboosting
    # clf_boosting = AdaBoostClassifier(base_estimator=base_est, n_estimators=50, algorithm='SAMME', random_state=0).fit(
    #     X_train, Y_train)
    # boosting_predict = clf_boosting.predict(X_pre)
    myAdaboosting=Enselearning.Adaboosting(X_train,Y_train,Indexs[::-1,:],C=C)
    myAdaboosting.fit()
    boosting_predict=myAdaboosting.predict(X_pre)
    MSE_AdaBoosting = (boosting_predict != Y_pre).mean()
    TP = sum([x == 1 and y == 1 for x, y in zip(Y_pre, boosting_predict)])
    TPR_AdaBoosting = TP / P
    FP = sum([x == -1 and y == 1 for x, y in zip(Y_pre, boosting_predict)])
    FPR_AdaBoosting = FP / N



    if MSE_ma>0.5:
        print(f'samplesize={samplesize},repea'
              f't={repeat},MSE_ma={MSE_ma},出问题了！')
    # ratio
    minloss = MA_method.get_minloss(X_pre, Y_pre)
    ratio_cl = hingeloss_cl / minloss
    ratio_ch = hingeloss_ch / minloss
    # ratio_ch2 = hingeloss_ch2 / minloss
    # ratio_ch3 = hingeloss_ch3 / minloss
    # ratio_ch4 = hingeloss_ch4 / minloss
    ratio_ma2 = hingeloss_ma2 / minloss
    ratio_ma = hingeloss_ma / minloss
    ratio_full = hingeloss_full / minloss
    ratio_unif = hingeloss_unif / minloss
    ratio_SCL = hingeloss_SCL / minloss
    ratio_SCH = hingeloss_SCH / minloss
    df = pd.DataFrame({'repeat': repeat,
                       'n': X_train.shape[0],
                       'p': X_train.shape[1],
                       'ratio_cl': ratio_cl,
                       'ratio_ch': ratio_ch,
                       'ratio_scl': ratio_SCL,
                       'ratio_sch': ratio_SCH,
                       # 'ratio_ch2': ratio_ch2,
                       # 'ratio_ch3': ratio_ch3,
                       # 'ratio_ch4': ratio_ch4,
                       'ratio_ma2': ratio_ma2,
                       'ratio_ma': ratio_ma,
                       'ratio_full': ratio_full,
                       'ratio_unif': ratio_unif,
                       'TPR_cl': TPR_cl,
                       'TPR_ch': TPR_ch,
                       # 'TPR_ch2': TPR_ch2,
                       # 'TPR_ch3': TPR_ch3,
                       # 'TPR_ch4': TPR_ch4,
                       'TPR_ma2': TPR_ma2,
                       'TPR_ma': TPR_ma,
                       'TPR_full': TPR_full,
                       'TPR_unif': TPR_unif,
                       'TPR_scl': TPR_SCL,
                       'TPR_sch': TPR_SCH,
                       'TPR_Bagging': TPR_Bagging,
                       'TPR_AdaBoosting': TPR_AdaBoosting,
                       'minloss': minloss,
                       'FPR_cl': FPR_cl,
                       'FPR_ch': FPR_ch,
                       # 'FPR_ch2': FPR_ch2,
                       # 'FPR_ch3': FPR_ch3,
                       # 'FPR_ch4': FPR_ch4,
                       'FPR_ma2': FPR_ma2,
                       'FPR_ma': FPR_ma,
                       'FPR_full': FPR_full,
                       'FPR_unif': FPR_unif,
                       'FPR_scl': FPR_SCL,
                       'FPR_sch': FPR_SCH,
                       'FPR_Bagging': FPR_Bagging,
                       'FPR_AdaBoosting': FPR_AdaBoosting,
                       'MSE_cl': MSE_cl,
                       'MSE_ch': MSE_ch,
                       'MSE_scl': MSE_SCL,
                       'MSE_sch': MSE_SCH,
                       # 'MSE_ch2': MSE_ch2,
                       # 'MSE_ch3': MSE_ch3,
                       # 'MSE_ch4': MSE_ch4,
                       'MSE_ma2': MSE_ma2,
                       'MSE_ma': MSE_ma,
                       'MSE_full': MSE_full,
                       'MSE_unif': MSE_unif,
                       'MSE_Bagging': MSE_Bagging,
                       'MSE_AdaBoosting': MSE_AdaBoosting,
                       'CL_select': CL_bestmodel_ind,
                       'CH_select': CH_bestmodel_ind
                       # 'minloss':minloss
                       # 'hingeloss_cl':hingeloss_cl,
                       # 'hingeloss_ch':hingeloss_ch,
                       # 'hingeloss_sbic':hingeloss_sbic,
                       # 'hingeloss_ma':hingeloss_ma
                       }, index=[0])

    # tidy results
    # dfH = pd.DataFrame(SVMICH_results, index=['SVMICH'])
    # dfL = pd.DataFrame(SVMICL_results, index=['SVMICL'])
    # dfSBIC = pd.DataFrame(SBIC_results, index=['SBIC'])

    if not os.path.exists(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\replication\\'):
        os.makedirs(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\replication\\')
    # dfH = pd.concat([dfH, dfL, dfSBIC], axis=0)
    df.to_csv(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\replication\\repeat={repeat}.csv', index=False)
    if not os.path.exists(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\Indexs\\'):
        os.makedirs(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\Indexs\\')
    pd.DataFrame(Indexs).to_csv(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\Indexs\\repeat={repeat}.csv', index=False)

    if not os.path.exists(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\weights\\'):
        os.makedirs(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\weights\\')
    weight_df.to_csv(f'..\\Results\\ismisspecification={ismisspecification}\\modeltype={modeltype}\\n={samplesize}_p={p}\\weights\\repeat={repeat}.csv', index=False)


    return df





def generate_data_fun(repeat, p, samplesize,ismisspecification=False):
    # generate the data and parameters setting
    # p is the dimension of beta
    # q is the number of beta!=0
    q = 50
    if ismisspecification:
        q=q+1
        p=p+1


    # replication = 200
    # samplesize = 200
    testsize = 10000
    Mn = int(samplesize ** (1/3))
    n = samplesize + testsize
    np.random.seed(1000 * repeat)

    # 均值和方差
    mu = np.zeros(p)
    mu[0:q] = 1
    Sigma2 = np.eye(p) *1.2-np.ones((p, p)) * (0.2)

    # 产生Y
    Y = np.random.binomial(n=1, p=0.5, size=n)
    Y[Y == 0] = -1


    X=np.random.multivariate_normal(mean=np.zeros(len(Sigma2)),cov=Sigma2,size=n)

    X[Y == 1] = X[Y == 1] + mu
    X[Y == -1] = X[Y == -1] - mu

    Y_bayes = np.array([np.sign(np.sum(x[0:q-1])) for x in X])
    Bayes_error = (Y_bayes != Y).mean()
    X_train = X[0:samplesize, :]
    Y_train = Y[0:samplesize]
    X_pre = X[samplesize:, :]
    Y_pre = Y[samplesize:]
    S_true = {0, 1, 2, 3}  # 真实模型的索引,在后面计算其实没啥用


    ###############################################
    #目的是要让其缺失一个重要变量，令真实模型不在里面
    if ismisspecification:
        inds=list(range(0,q-1))
        inds.extend(list(range(q,X.shape[1])))
    else:
        inds=list(range(0,X.shape[1]))
    ###############################################

    return X_train[:,inds], Y_train, X_pre[:,inds], Y_pre, S_true, Mn,Bayes_error

def generate_data_fun2(repeat, p, samplesize,ismisspecification=True):
    # 产生数据
    # p = 10
    q = 4
    if ismisspecification:
        q=q+1
        p=p+1


    testsize = 10000
    Mn = int(samplesize ** (1/3))
    n = samplesize + testsize
    np.random.seed(1000 * repeat)

    beta=np.zeros(p)
    beta[0:q]=2

    #genereate X
    Sigma2=np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            Sigma2[i,j]=0.4**np.abs(i-j)
    # R = cholesky(Sigma2)
    # X = np.dot(np.random.randn(n, p+1), R)

    X=np.random.multivariate_normal(mean=np.zeros(len(Sigma2)),cov=Sigma2,size=n)


    #generate Y
    Y=np.random.binomial(n=1, p=stat.norm.cdf(np.dot(X, beta)), size=n)
    Y[Y == 0] = -1

    beta2=np.array([0.8,0.8,0.8,0.8,0.8]*30)[0:q]
    Y_bayes = np.array([np.sign(np.dot(x[0:q],beta2) ) for x in X])
    Bayes_error = (Y_bayes != Y).mean()
    X_train = X[0:samplesize, :]
    Y_train = Y[0:samplesize]
    X_pre = X[samplesize:, :]
    Y_pre = Y[samplesize:]
    S_true = {0, 1, 2, 3}  # 真实模型的索引
    ###############################################
    if ismisspecification:
        inds=list(range(0,q-1))
        inds.extend(list(range(q,X.shape[1])))
    else:
        inds=list(range(0,X.shape[1]))
    #3##############################################
    return X_train[:,inds], Y_train, X_pre[:,inds], Y_pre, S_true, Mn,Bayes_error


def Tidy(samplesize, p,modeltype):
    folder = os.path.abspath(f'..\\Results\\modeltype={modeltype}\\n={samplesize}_p={p}\\replication\\')
    DF = pd.DataFrame()
    for file in os.listdir(folder):
        df = pd.read_csv(folder + '\\' + file, index_col=0)
        DF = pd.concat([DF, df])
    DF = DF.groupby(level=0).mean()
    if not os.path.exists(f'..\\Results\\modeltype={modeltype}\\n={samplesize}_p={p}\\tidy_data\\'):
        os.makedirs(f'..\\Results\\modeltype={modeltype}\\n={samplesize}_p={p}\\tidy_data\\')
    DF.to_csv(f'..\\Results\\modeltype={modeltype}\\n={samplesize}_p={p}\\tidy_data\\meandata.csv', index=True)



def work_fun(args):
    samplesize, p, modeltype,ismisspecification, repeat=args
    print(f'modeltpe={modeltype}_ismisspecification={ismisspecification}_repeat={repeat} start!')
    if modeltype==1:
        X_train, Y_train, X_pre, Y_pre, S_true, Mn,Bayes_error = generate_data_fun(repeat, p, samplesize,ismisspecification)
    elif modeltype==2:
        X_train, Y_train, X_pre, Y_pre, S_true, Mn,Bayes_error = generate_data_fun2(repeat, p, samplesize,ismisspecification)
    else:
        print('modeltype is wrong!')
        return

    print(f'Bayes_error:{Bayes_error}')
    res = main(repeat, X_train, Y_train, X_pre, Y_pre, Mn, S_true,modeltype=modeltype,ismisspecification=ismisspecification)
    print(f'modeltpe={modeltype}_ismisspecification={ismisspecification}_repeat={repeat} finished')


class worker(Process):
    def __init__(self, mission_queue):
        Process.__init__(self)
        self.mission_queue = mission_queue

    def run(self):
        while self.mission_queue.empty() == False:
           work_fun(self.mission_queue.get())

if __name__=='__main__1':
    work_fun((20,100, 2,True, 6))

if __name__=='__main_1_':
    modelsettings=[(200,1000),(500,1000),(800,1000),(1000,1000),(1500,1000)]#[(200,1000),(400,1000),(800,1000),(1500,1000)]
    Figures=figure_file.SVMMAfigure(modelsettings, modeltype=1,ismisspecification=True)
    Figures.tidy_fun()
    Figures.figure_fun2()
#
#
if __name__ == '__main__':
    mission_queue = Queue()
    modeltypeset=[2]

    modelsettings =[(100,1000),(200,1000),(300,1000)]#,(400,1000)]#]#(200,400),,(400,1000)]#,(300,1000),(400,1000)]#,(300,1000),(400,1000)][(100,1000)]
    ismisset=[True,False]
    for n_p in modelsettings:
        for modeltype in modeltypeset:
            for repeat in range(20):
                for ismisspecification in ismisset:
                    mission_queue.put((n_p[0],n_p[1],modeltype,ismisspecification,repeat))

    # # 创建n个进程类，并开启
    Workers=[]
    for i in range(10):
        workrobot = worker(mission_queue)
        workrobot.start()
        Workers.append(workrobot)

    for worker in Workers:
        worker.join()

    #generate figures
    for modeltype in modeltypeset:
        for ismisspecification in ismisset:
            Figures=figure_file.SVMMAfigure(modelsettings, modeltype=modeltype,ismisspecification=ismisspecification)
            Figures.tidy_fun()
            Figures.figure_fun()

    print('The program is finished!')
    # # Tidy(samplesize, p, modeltype=1)
    # # Tidy(samplesize, p, modeltype=2)
    # # Tidy的相关路径需要调整一下