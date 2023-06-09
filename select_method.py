import numpy as np


def Select_Evaluation(bestmodel_ind, X_pre, Y_pre, Models, Indexs):
    # 该函数用于对选中模型的指标评估计算
    p = Indexs.shape[1]
    results = dict()
    results['bestmodel'] = bestmodel_ind
    clfbest = Models[results['bestmodel']]
    Y_res = clfbest.predict(X_pre[:, Indexs[results['bestmodel']]])
    results['testerror'] = np.array((Y_res != Y_pre)).mean()
    # 得到估计模型的变量序号
    return results

class SVMICL():
    def __init__(self,HingesLoss,Indexs,Models,samplesize):
        self.HingesLoss=HingesLoss
        self.Indexs=Indexs
        self.samplesize=samplesize
        self.Models=Models
        self.bestmodel_ind=None
        self.SVMICL_scores=[]

    def get_scores(self):
        SVMICL_scores=[]
        for hingeloss,index in zip(self.HingesLoss,self.Indexs):
            SVMICL_scores.append(hingeloss + sum(index) * np.log(self.samplesize))
        self.SVMICL_scores=SVMICL_scores

    def get_bestmodel(self,m=1):
        # model selection
        if not self.SVMICL_scores:
            self.get_scores()

        inds=np.argsort(self.SVMICL_scores)
        self.bestmodel_ind = inds[0:m]#self.SVMICH_scores.index(min(self.SVMICH_scores))
        return self.bestmodel_ind

    def do_evaluation(self,X_pre,Y_pre):
        if not self.SVMICL_scores:
            self.get_scores()
        if not self.bestmodel_ind:
            self.get_bestmodel()
        # evaluation
        return Select_Evaluation(self.bestmodel_ind, X_pre, Y_pre, self.Models, self.Indexs)



class SVMICH():
    def __init__(self,HingesLoss,Indexs,Models,samplesize,Ln):
        self.HingesLoss=HingesLoss
        self.Indexs=Indexs
        self.samplesize=samplesize
        self.Models=Models
        self.bestmodel_ind=None
        self.Ln=Ln
        self.SVMICH_scores=[]

    def get_scores(self):
        SVMICH_scores=[]
        for hingeloss,index in zip(self.HingesLoss,self.Indexs):
            SVMICH_scores.append(hingeloss + self.Ln * sum(index) * np.log(self.samplesize))
        self.SVMICH_scores=SVMICH_scores

    def get_bestmodel(self,m=1):
        # model selection
        if not self.SVMICH_scores:
            self.get_scores()

        inds=np.argsort(self.SVMICH_scores)
        self.bestmodel_ind = inds[0:m]#self.SVMICH_scores.index(min(self.SVMICH_scores))
        return self.bestmodel_ind

    def do_evaluation(self,X_pre,Y_pre):
        if not self.SVMICH_scores:
            self.get_scores()
        if not self.bestmodel_ind:
            self.get_bestmodel()
        # evaluation
        return Select_Evaluation(self.bestmodel_ind, X_pre, Y_pre, self.Models, self.Indexs)
