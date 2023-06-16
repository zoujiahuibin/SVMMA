'''
This script is used to draw learning curves
'''

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random

from codes.delete import test_bag, test_ada, test_ma
import main


def read_data():
    samplesize, p,  repeat = int(208*0.8),60,1
    data = pd.read_csv('sonar.txt', header=None, index_col=None)
    data.columns = [f'x{i}' for i in range(1, data.shape[1])] + ['y']
    data['y'] = data['y'].replace(' R', 1).replace(' M', -1)

    X = np.array(data.drop('y', axis=1))
    X = preprocessing.scale(X)
    y = np.array(data['y'])
    random.seed(22345)#2
    X_train, X_pre, Y_train, Y_pre = train_test_split(X, y, train_size=samplesize, random_state=repeat,
                                                      shuffle=True, stratify=y)
    return X_train, Y_train,X_pre, Y_pre



if __name__ == '__main__':
    num_of_candidate=60
    X_train, Y_train,X_pre, Y_pre=read_data()
    Indexs = main.Indexs_fun_lasso(X_train, Y_train, k=num_of_candidate)
    MSE_ma,MSPE_ma,w_ma= test_ma.learn_curve(X_train, Y_train, X_pre, Y_pre, Indexs)
    MSE_bag,MSPE_bag= test_bag.learn_curve(X_train, Y_train, X_pre, Y_pre, Indexs)
    MSE_ada,MSPE_ada,w_ada= test_ada.learn_curve(X_train, Y_train, X_pre, Y_pre, Indexs, Iter_max=num_of_candidate)

    Res=pd.DataFrame({'MSE_ma':MSE_ma,'MSPE_ma':MSPE_ma,'MSE_bag':MSE_bag,'MSPE_bag':MSPE_bag,'MSE_ada':MSE_ada,'MSPE_ada':MSPE_ada})
    dfw=pd.DataFrame({'w_ma':w_ma,'w_ada':w_ada})
    if not os.path.exists(f'..\\Results\\Learn_curves\\'):
        os.makedirs(f'..\\Results\\Learn_curves\\')
    Res.to_csv(f'..\\Results\\Learn_curves\\Results.csv', index=False)
    dfw.to_csv(f'..\\Results\\Learn_curves\\dfw.csv', index=False)


    Res = pd.read_csv('..\\Results\\Learn_curves\\Results.csv')
    plt.figure()
    plt.plot(range(1,len(Res['MSE_ma'])+1),Res['MSE_ma'],'-',label='SVMMA-Train',color='blue',lw=1,markersize=7)
    plt.plot(range(1,len(Res['MSE_ma'])+1),Res['MSPE_ma'],'--',label='SVMMA-Test',color='orange',lw=1,markersize=7)
    plt.plot(range(1,len(Res['MSE_ma'])+1),Res['MSE_bag'],'-o',label='Bag-Train',color='blue',lw=1,markersize=3)
    plt.plot(range(1,len(Res['MSE_ma'])+1),Res['MSPE_bag'],'--o',label='Bag-Test',color='orange',lw=1,markersize=3)
    plt.plot(range(1,len(Res['MSE_ma'])+1),Res['MSE_ada'],'-3',label='Ada-Train',color='blue',lw=1,markersize=7)
    plt.plot(range(1,len(Res['MSE_ma'])+1),Res['MSPE_ada'],'--3',label='Ada-Test',color='orange',lw=1,markersize=7)
    plt.legend(loc=3, prop={'size': 7}, ncol=2)
    plt.ylabel('Error rate')
    plt.xlabel('No. of candidate models')
    plt.ylim((0,0.6))
    fig = plt.gcf()
    fig.set_size_inches(8, 3)
    if not os.path.exists(f'../figures/compare'):
        os.makedirs(f'../figures/compare')
    plt.savefig(f'../figures/compare/learn_curves.png', dpi=250, bbox_inches='tight', pad_inches=0.1)


