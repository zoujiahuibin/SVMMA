'''
the comparators：
    SVMIC_L with Ln=1
    SVMIC_H with Ln=\sqrt{\log(n)}
    FULL model: the biggest submodel in candidate model set
    uniform model
    MA with sum(w)=1,
'''




import os
import pandas as pd
from multiprocessing import Process, Queue
import ma_method
import figure_file
from tools_function import *
from sklearn.model_selection import train_test_split
import time
from sklearn import preprocessing
import Enselearning


def main(repeat, X_train, Y_train, X_pre, Y_pre, Mn, S_true, datatype, modeltype):
    # parameter prepration
    samplesize = X_train.shape[0]
    p = X_train.shape[1]
    lambda_n = samplesize ** (-1)
    C = (samplesize * lambda_n) ** (-1)

    # candidate models setting up
    Mn=30
    Indexs = Indexs_fun_lasso(X_train, Y_train,k=Mn)
    # estimate for candidate model
    Models = []
    Betas = []
    HingesLoss = []
    for index in Indexs:
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
    CH_method4 = select_method.SVMICH(HingesLoss, Indexs, Models, samplesize, Ln=samplesize**(-1/3))
    w_CL = np.zeros(Indexs.shape[0])
    w_CH = np.zeros(Indexs.shape[0])
    w_CH4 = np.zeros(Indexs.shape[0])
    w_full=np.zeros(Indexs.shape[0])
    CL_bestmodel_ind = CL_method.get_bestmodel()
    w_CL[CL_bestmodel_ind] = 1
    CH_bestmodel_ind = CH_method.get_bestmodel()
    w_CH[CH_bestmodel_ind] = 1
    CH_bestmodel_ind4 = CH_method4.get_bestmodel()
    w_CH4[CH_bestmodel_ind4]=1
    w_full[-1] = 1
    w_unif=np.ones(Indexs.shape[0])/Indexs.shape[0]


    MA_inds=range(len(Indexs))
    MA_Indexs=Indexs[MA_inds,:]
    MABetas=[Betas[i] for i in MA_inds]
    MA_method = ma_method.SVMMA(X_train, Y_train, MA_Indexs, samplesize, MABetas, Jn=5, C=C)
    MA_method2 = ma_method.SVMMA2(X_train, Y_train, MA_Indexs, samplesize, MABetas, Jn=5, C=C)
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



    # evaluation
    TPR_full, FPR_full, MSE_full, hingeloss_full = ma_method.MA_Evaluation(w_full, X_pre, Y_pre, Models, Indexs)
    TPR_cl, FPR_cl, MSE_cl, hingeloss_cl = ma_method.MA_Evaluation(w_CL, X_pre, Y_pre, Models, Indexs)
    TPR_ch, FPR_ch, MSE_ch, hingeloss_ch = ma_method.MA_Evaluation(w_CH, X_pre, Y_pre, Models, Indexs)
    TPR_ch4, FPR_ch4, MSE_ch4, hingeloss_ch4 = ma_method.MA_Evaluation(w_CH4, X_pre, Y_pre, Models, Indexs)
    TPR_ma, FPR_ma, MSE_ma, hingeloss_ma = ma_method.MA_Evaluation(w_ma_hat, X_pre, Y_pre, Models, Indexs)
    TPR_ma2, FPR_ma2, MSE_ma2, hingeloss_ma2 = ma_method.MA_Evaluation(w_ma_hat2, X_pre, Y_pre, Models, Indexs)
    TPR_unif, FPR_unif, MSE_unif, hingeloss_unif = ma_method.MA_Evaluation(w_unif, X_pre, Y_pre, Models, Indexs)
    TPR_SCL, FPR_SCL, MSE_SCL, hingeloss_SCL = ma_method.MA_Evaluation(w_SCL, X_pre, Y_pre, Models, Indexs)
    TPR_SCH, FPR_SCH, MSE_SCH, hingeloss_SCH = ma_method.MA_Evaluation(w_SCH, X_pre, Y_pre, Models, Indexs)



 # implement bagging
    myBagging=Enselearning.Bagging(X_train,Y_train,Models,Indexs)
    myBagging.get_weights(type='unif')
    bagging_predict=myBagging.predict(X_pre)
    MSE_Bagging = (bagging_predict != Y_pre).mean()
    P=sum(Y_pre==1)
    N=sum(Y_pre==-1)
    TP=sum([x==1 and y==1 for x,y in zip(Y_pre,bagging_predict)])
    TPR_Bagging=TP/P
    FP=sum([x==-1 and y==1 for x,y in zip(Y_pre,bagging_predict)])
    FPR_Bagging=FP/N



    # implement adaboosting
    myAdaboosting=Enselearning.Adaboosting(X_train,Y_train,Indexs,C=C)
    myAdaboosting.fit(Mn)
    boosting_predict=myAdaboosting.predict(X_pre)
    MSE_AdaBoosting = (boosting_predict != Y_pre).mean()
    TP=sum([x==1 and y==1 for x,y in zip(Y_pre,boosting_predict)])
    TPR_AdaBoosting = TP / P
    FP = sum([x == -1 and y == 1 for x, y in zip(Y_pre, boosting_predict)])
    FPR_AdaBoosting = FP / N
    adaw = np.zeros(Indexs.shape[0])
    adaw[0:len(myAdaboosting.w)] = myAdaboosting.w


    weight_df = pd.DataFrame({'w_CL': w_CL,
                              'w_CH': w_CH,
                              'w_SCL': w_SCL,
                              'w_SCH': w_SCH,
                              # 'w_CH4': w_CH4,
                              'w_ma': w_ma_hat,
                              'w_ma2': w_ma_hat2,
                              'w_bag':myBagging.w.T[0],
                              'w_ada':adaw})



    if MSE_ma>0.5:
        print(f'samplesize={samplesize},repea'
              f't={repeat},MSE_ma={MSE_ma},出问题了！')
    # ratio
    minloss = MA_method.get_minloss(X_pre, Y_pre)
    ratio_cl = hingeloss_cl / minloss
    ratio_ch = hingeloss_ch / minloss
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
                       'ratio_ma2': ratio_ma2,
                       'ratio_ma': ratio_ma,
                       'ratio_full': ratio_full,
                       'ratio_unif': ratio_unif,
                       'TPR_cl': TPR_cl,
                       'TPR_ch': TPR_ch,
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
                       'MSE_ma2': MSE_ma2,
                       'MSE_ma': MSE_ma,
                       'MSE_full': MSE_full,
                       'MSE_unif': MSE_unif,
                       'MSE_Bagging': MSE_Bagging,
                       'MSE_AdaBoosting': MSE_AdaBoosting,
                       'CL_select': CL_bestmodel_ind,
                       'CH_select': CH_bestmodel_ind
                       }, index=[0])

    # tidy results
    if not os.path.exists(
            f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\replication\\'):
        os.makedirs(f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\replication\\')
    df.to_csv(
        f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\replication\\repeat={repeat}.csv',
        index=False)
    if not os.path.exists(f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\Indexs\\'):
        os.makedirs(f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\Indexs\\')
    pd.DataFrame(Indexs).to_csv(
        f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\Indexs\\repeat={repeat}.csv',
        index=False)

    if not os.path.exists(f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\weights\\'):
        os.makedirs(f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\weights\\')
    weight_df.to_csv(
        f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\weights\\repeat={repeat}.csv',
        index=False)

    return df





def Tidy(samplesize, p, datatype, modeltype):
    folder = os.path.abspath(
        f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\replication\\')
    DF = pd.DataFrame()
    for file in os.listdir(folder):
        df = pd.read_csv(folder + '\\' + file, index_col=0)
        DF = pd.concat([DF, df])
    DF = DF.groupby(level=0).mean()
    if not os.path.exists(
            f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\tidy_data\\'):
        os.makedirs(f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\tidy_data\\')
    DF.to_csv(f'..\\Results\\datatype={datatype}\\modeltype={modeltype}\\n={samplesize}_p={p}\\tidy_data\\meandata.csv',
              index=True)


def work_fun(args):
    samplesize, p, modeltype, repeat, datatype = args
    if datatype == 'realdata':
        mydata = pd.read_csv('ionosphere.data', sep=',', header=None)
        mydata.drop(1, axis=1, inplace=True)

        columns = [f'x{i + 1}' for i in range(mydata.shape[1])]
        columns[-1] = 'y'
        mydata.columns = columns
        mydata['y'] = mydata['y'].replace('g', 1).replace('b', -1)

        X = np.array(mydata.drop('y', axis=1))
        X=preprocessing.scale(X)
        y = np.array(mydata['y'])


        X_train, X_pre, Y_train, Y_pre = train_test_split(X, y, train_size=samplesize, random_state=repeat,
                                                          shuffle=True, stratify=y)
        S_true = set(range(X_train.shape[1]))
        Mn = int(X_train.shape[0] ** (1 / 3))
        Bayes_error = 'unknown'
    else:
        print('datatype is wrong!')
        return

    print(f'Bayes_error:{Bayes_error}')
    res = main(repeat, X_train, Y_train, X_pre, Y_pre, Mn, S_true, datatype, modeltype=modeltype)
    print(f'modeltpe={modeltype}_repeat={repeat} finished')


class worker(Process):
    def __init__(self, mission_queue):
        Process.__init__(self)
        self.mission_queue = mission_queue

    def run(self):
        while self.mission_queue.empty() == False:
            work_fun(self.mission_queue.get())



if __name__ == '__main__':
    # for realdata
    start_time=time.time()
    N = 351
    p = 33
    modelsettings=[(int(N*i),p) for i in [0.4,0.5,0.6,0.7,0.8]]


    mission_queue = Queue()
    for n_p in modelsettings:  # ,(1500,1000)]:
        for modeltype in [1]:
            for repeat in range(200):
                for datatype in ['realdata']:
                    mission_queue.put((n_p[0], n_p[1], modeltype, repeat, datatype))


    Workers = []
    for i in range(10):
        workrobot = worker(mission_queue)
        workrobot.start()
        Workers.append(workrobot)

    for worker in Workers:
        worker.join()

    for i in [1]:
        for datatype in ['realdata']:
            Figures = figure_file.SVMMAfigure(modelsettings, datatype, modeltype=i)
            Figures.tidy_fun()
            Figures.figure_fun2()

    # 生成平均值的表格
    for i in [1]:
        for datatype in ['realdata']:
            for samplesize, p in modelsettings:
                Tidy(samplesize, p, datatype, modeltype=i)
        Figures=figure_file.SVMMAfigure(modelsettings,datatype='realdata', modeltype=i)
        Figures.tidy_fun()
        Figures.figure_fun2()
    print('the time used is:', time.time()-start_time)


