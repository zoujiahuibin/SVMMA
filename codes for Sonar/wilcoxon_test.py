

import os
import pandas as pd
from scipy import stats


if __name__ == '__main__':
    nset = [83, 104, 124, 145, 166]
    pset = [60] * len(nset)
    Methods = ['cl', 'ch', 'scl', 'sch', 'unif', 'Bagging', 'AdaBoosting']
    Test_DF = pd.DataFrame()
    for n, p in zip(nset, pset):
        DF = pd.DataFrame()

        for file in os.listdir(f'../Results/datatype=realdata/modeltype=1/n={n}_p={p}/replication'):
            df = pd.read_csv(f'../Results/datatype=realdata/modeltype=1/n={n}_p={p}/replication/{file}')
            DF = pd.concat((DF, df), axis=0)
        wilcox = dict()
        pvalue = dict()
        wilcox['n'] = n
        pvalue['n'] = n
        for method in Methods:
            wilcox[f'{method}/ma'], pvalue[f'{method}/ma'] = stats.wilcoxon(DF[f'MSE_{method}'], DF[f'MSE_ma'],
                                                                            zero_method='wilcox', alternative='greater')
        Test_DF = pd.concat((Test_DF, pd.DataFrame(wilcox, index=['Wilcox']), pd.DataFrame(pvalue, index=['p-value'])))
    if not os.path.exists(f'../Results/Wilcox_test/'):
        os.makedirs(f'../Results/Wilcox_test/')

    Test_DF.to_excel(f'../Results/Wilcox_test/Wilcox_test.xlsx')
    print('finish！')
