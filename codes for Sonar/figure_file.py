import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class SVMMAfigure():
    def __init__(self, modelsettings, datatype, modeltype):
        self.modelsettings = modelsettings
        self.modeltype = modeltype
        self.datatype = datatype
        self.linewidth = 0.7
        self.markersize = '4'
        # self.myaddress = os.path.abspath(os.getcwd() + '/../')
        self.summary_dir = os.path.abspath(
            '..\\figures\\datatype={1}\\modeltype={0}\\summary\\'.format(modeltype, datatype))
        self.folder_figure = os.path.abspath('..\\figures\\datatype={1}\\modeltype={0}\\'.format(modeltype, datatype))
        methlist = ['cl', 'ch', 'scl', 'sch','ma','unif']
        methlist2 = ['cl', 'ch', 'scl', 'sch','ma','unif','Bagging','AdaBoosting']
        self.ratio_varlist = ['ratio_' + nam for nam in methlist]
        self.MSE_varlist = ['MSE_' + nam for nam in methlist2]
        self.NMSE_varlist = ['NMSE_' + nam for nam in methlist]
        self.TPR_varlist = ['TPR_' + nam for nam in methlist]
        self.FPR_varlist = ['FPR_' + nam for nam in methlist]
        self.x_labels = [r'$SVMICL$', r'$SVMICH$',r'$SCL$','r$SCH$', r'$SVMMA$', r'$UNIF$']
        self.stype_list = ['--+', '-*', '-.o', ':v', '--^', '-.D', '--1', '-s']


    def tidy_fun(self):
        # samplesizeset = self.samplesizeset
        modeltype = self.modeltype

        # collect the data
        df = pd.DataFrame()
        for n, p in self.modelsettings:
            folder = os.path.abspath(
                '..\\Results\\datatype={3}\\modeltype={0}\\n={1}_p={2}\\replication\\'.format(modeltype, n, p,
                                                                                              self.datatype))
            for file in os.listdir(folder):
                tempfile = pd.read_csv(folder + '\\' + file)
                df = pd.concat([df, tempfile])
        folder2 = self.summary_dir
        if not os.path.exists(folder2):
            os.makedirs(folder2)
        df.to_csv(folder2 + '/alldata.csv', index=False)
        df = pd.read_csv(folder2 + '/alldata.csv')
        self.alldata = df.copy()

        df1 = df.melt(id_vars=['n', 'p', 'repeat'], var_name=['methods'])
        df2 = pd.pivot_table(df1, index=['n', 'p', 'repeat'], columns=['methods'])
        # average
        df3 = df2.groupby(level=['n', 'p']).mean()
        # change the names of columns
        df3.columns = [i[1] for i in df3.columns]
        df4 = df3.reset_index(level=['n', 'p'])
        df4.to_csv(folder2 + '/meandata.csv', index=False)
        self.meandata = df4.copy()

    def figure_fun(self):
        # modelsettings = self.modelsettings
        folder_figure = self.folder_figure
        if not os.path.exists(folder_figure):
            os.makedirs(folder_figure)
        df = self.meandata.copy()
        # the line chart of MSE, based on meandata.csv
        stype_list = self.stype_list

        # optimal property, based on meandata.csv
        R2set = np.unique(df['p'])
        self.R2set = R2set
        ratio_list = self.ratio_varlist
        for p in R2set:
            plt.figure()
            for var, stype in zip(ratio_list, stype_list):
                df_temp = df[df['p'] == p]
                plt.plot(df_temp['n'], df_temp[var], stype, label=var[6:].upper().replace('SCL','S1').replace('SCH','S2').replace('CL','SVMICL').replace('CH','SVMICH').replace('MA','SVMMA').replace('S1','SCL').replace('S2','SCH').replace('BAGGING','BAG').replace('ADABOOSTING','ADA'), lw=self.linewidth,
                         markersize=self.markersize)
            plt.title('NHL comparisons')
            plt.xlabel('Size of the training sample')
            plt.legend(loc=2, prop={'size': 7}, ncol=2)
            fig = plt.gcf()
            fig.set_size_inches(3.75, 3)
            plt.savefig(folder_figure + f'/optimality_p={p}.png', dpi=250, bbox_inches='tight', pad_inches=0.1)

        # the boxplot of MSE
        df = self.alldata.copy()
        for p in R2set:
            plt.figure()
            df_temp = df[df['p'] == p]
            df_temp = df_temp.dropna(axis=0)
            plt.boxplot(df_temp[self.MSE_varlist].T)
            plt.xticks(range(1, len(self.MSE_varlist) + 1), self.x_labels)
            plt.axhline(np.median(df_temp['MSE_ma'], ), color="red", linestyle='--')
            plt.title('The boxplot of error rate')
            plt.savefig(folder_figure + f'/The_boxplot_of_MSE_p={p}.png', dpi=250, bbox_inches='tight', pad_inches=0.1)

        # the line chart of NMS，x axis represents the training size
        varname_list = ['n', 'p', 'repeat']
        varname_list.extend(self.MSE_varlist)
        for p in R2set:
            df_temp = df.loc[df['p'] == p, varname_list]
            df1 = df_temp.melt(id_vars=['n', 'repeat'], var_name=['methods'])
            # generate mutiIndex table
            df2 = pd.pivot_table(df1, index=['n', 'repeat'], columns=['methods'])
            df3 = df2.groupby(level=['n']).mean()
            df3.columns = [i[1] for i in df3.columns]
            df3 = df3[self.MSE_varlist]

            plt.figure()
            for var, stype in zip(self.MSE_varlist, stype_list):
                plt.plot(df3[var], stype, label=var[4:].upper().replace('SCL','S1').replace('SCH','S2').replace('CL','SVMICL').replace('CH','SVMICH').replace('MA','SVMMA').replace('S1','SCL').replace('S2','SCH').replace('BAGGING','BAG').replace('ADABOOSTING','ADA'), lw=self.linewidth, markersize=self.markersize)
            plt.title('ER')
            plt.xlabel('Size of the training sample')
            plt.legend(loc=2, prop={'size': 7},ncol=2)
            fig = plt.gcf()
            fig.set_size_inches(3.75, 3)
            plt.savefig(folder_figure + f'/The_mean_of_NMSE_realdata_p={p}.png', dpi=250, bbox_inches='tight',
                        pad_inches=0.1)

    def figure_fun2(self):
        # modelsettings = self.modelsettings
        folder_figure = self.folder_figure
        if not os.path.exists(folder_figure):
            os.makedirs(folder_figure)
        df = self.meandata.copy()
        # the line chart of MSE, based on meandata.csv
        stype_list = self.stype_list

        # optimal property, based on meandata.csv
        R2set = np.unique(df['p'])
        # 应当把n也加上
        self.R2set = R2set
        ratio_list = self.ratio_varlist
        for p in R2set:
            plt.figure()
            for var, stype in zip(ratio_list, stype_list):
                df_temp = df[df['p'] == p]
                plt.plot(df_temp['n'], df_temp[var], stype, label=var[6:].upper().replace('SCL','S1').replace('SCH','S2').replace('CL','SVMICL').replace('CH','SVMICH').replace('MA','SVMMA').replace('S1','SCL').replace('S2','SCH').replace('BAGGING','BAG').replace('ADABOOSTING','ADA'), lw=self.linewidth,
                         markersize=self.markersize)
                plt.xticks(np.array([83,104,124,145,166]),['40%','50%','60%','70%','80%'])
            plt.title('NHL comparisons')
            plt.xlabel('Size of the training sample')
            plt.legend(loc=2, prop={'size': 7},ncol=2)
            fig = plt.gcf()
            fig.set_size_inches(3.75, 3)
            plt.savefig(folder_figure + f'/optimality_p={p}.png', dpi=250,
                        bbox_inches='tight', pad_inches=0.1)



        # the line chart of NMS，x axis represents the training size
        varname_list = ['n', 'p', 'repeat']
        varname_list.extend(self.MSE_varlist)
        df = self.alldata.copy()
        for p in R2set:
            df_temp = df.loc[df['p'] == p, varname_list]
            df1 = df_temp.melt(id_vars=['n', 'repeat'], var_name=['methods'])
            # generate mutiIndex table
            df2 = pd.pivot_table(df1, index=['n', 'repeat'], columns=['methods'])
            df3 = df2.groupby(level=['n']).mean()
            df3.columns = [i[1] for i in df3.columns]
            df3 = df3[self.MSE_varlist]

            plt.figure()
            for var, stype in zip(self.MSE_varlist, stype_list):
                plt.plot(df3[var], stype, label=var[4:].upper().replace('SCL','S1').replace('SCH','S2').replace('CL','SVMICL').replace('CH','SVMICH').replace('MA','SVMMA').replace('S1','SCL').replace('S2','SCH').replace('BAGGING','BAG').replace('ADABOOSTING','ADA'), lw=self.linewidth, markersize=self.markersize)
                plt.xticks(np.array([83,104,124,145,166]),['40%','50%','60%','70%','80%'])
            plt.title('ER comparisons')
            plt.xlabel('Size of the training sample')
            plt.legend(loc=2, prop={'size': 7},ncol=2)
            fig = plt.gcf()
            fig.set_size_inches(3.75, 3)
            plt.savefig(folder_figure + f'/The_mean_of_NMSE_p={p}.png',
                        dpi=250, bbox_inches='tight', pad_inches=0.1)

        varname_list = ['n', 'p', 'repeat']
        varname_list.extend(self.TPR_varlist)
        for p in R2set:
            df_temp = df.loc[df['p'] == p, varname_list]
            df1 = df_temp.melt(id_vars=['n', 'repeat'], var_name=['methods'])
            # generate mutiIndex table
            df2 = pd.pivot_table(df1, index=['n', 'repeat'], columns=['methods'])
            df3 = df2.groupby(level=['n']).mean()
            df3.columns = [i[1] for i in df3.columns]
            df3 = df3[self.TPR_varlist]

            plt.figure()
            for var, stype in zip(self.TPR_varlist, stype_list):
                plt.plot(df3[var], stype, label=var[4:].upper().replace('SCL','S1').replace('SCH','S2').replace('CL','SVMICL').replace('CH','SVMICH').replace('MA','SVMMA').replace('S1','SCL').replace('S2','SCH').replace('BAGGING','BAG').replace('ADABOOSTING','ADA'), lw=self.linewidth, markersize=self.markersize)
            plt.title('TPR comparisons')
            plt.xlabel('Size of the training sample')
            plt.legend(loc=2, prop={'size': 7},ncol=2)
            fig = plt.gcf()
            fig.set_size_inches(3.75, 3)
            plt.savefig(folder_figure + f'/The_mean_of_TPR_p={p}.png',
                        dpi=250, bbox_inches='tight', pad_inches=0.1)

            varname_list = ['n', 'p', 'repeat']
            varname_list.extend(self.FPR_varlist)
            for p in R2set:
                df_temp = df.loc[df['p'] == p, varname_list]
                df1 = df_temp.melt(id_vars=['n', 'repeat'], var_name=['methods'])
                # generate mutiIndex table
                df2 = pd.pivot_table(df1, index=['n', 'repeat'], columns=['methods'])
                df3 = df2.groupby(level=['n']).mean()
                df3.columns = [i[1] for i in df3.columns]
                df3 = df3[self.FPR_varlist]

                plt.figure()
                for var, stype in zip(self.FPR_varlist, stype_list):
                    plt.plot(df3[var], stype, label=var[4:].upper().replace('SCL','S1').replace('SCH','S2').replace('CL','SVMICL').replace('CH','SVMICH').replace('MA','SVMMA').replace('S1','SCL').replace('S2','SCH').replace('BAGGING','BAG').replace('ADABOOSTING','ADA'), lw=self.linewidth, markersize=self.markersize)
                plt.title('FPR comparisons')
                plt.xlabel('Size of the training sample')
                plt.legend(loc=2, prop={'size': 7},ncol=2)
                fig = plt.gcf()
                fig.set_size_inches(3.75, 3)
                plt.savefig(folder_figure + f'/The_mean_of_FPR_p={p}.png',
                            dpi=250, bbox_inches='tight', pad_inches=0.1)

    def correct_w_figure(self):
        # the line chart of correct_weight
        df = self.meandata.copy()
        plt.figure()
        R2set = [0.5, 0.7]
        nset = np.array([100, 300, 500])
        fun = lambda x: x in nset
        inds = [fun(x) for x in df['n']]
        df = df.iloc[inds, :]
        types = self.stype_list[0:3]
        for R_2, type in zip(R2set, types):
            df_temp = df[df['R_2'] == R_2]
            plt.plot(df_temp['samplesize'], df_temp["sum_corweights"], type, label=r'$R^2$={}'.format(R_2),
                     lw=self.linewidth,
                     markersize=self.markersize)
        # plt.title(r"The sum of correct model's weights when $R^2$={}".format(R_2))
        plt.title(r'$\sin(\cdot)$' + ', ' + self.modeltype)
        plt.xlabel('Size of the training sample')
        plt.legend(loc=2, prop={'size': 7},ncol=2)
        fig = plt.gcf()
        fig.set_size_inches(3.75, 3)
        plt.savefig(self.folder_figure + '/sumweights_R2={}.png'.format(str(R_2).replace('.', 'd')),
                    bbox_inches='tight', pad_inches=0.1, dpi=250)

    def thm3_figure(self):
        df = self.meandata.copy()
        R2set = [0.5, 0.7]
        nset = np.array([100, 300, 500])
        fun = lambda x: x in nset
        inds = [fun(x) for x in df['samplesize']]
        df = df.iloc[inds, :]
        types = self.stype_list[0:3]
        plt.figure()
        for R_2, type in zip(R2set, types):
            df_temp = df[df['R_2'] == R_2]
            plt.plot(df_temp['samplesize'], df_temp["thm3_value"], type, lw=self.linewidth,
                     label=r'$R^2$={}'.format(R_2),
                     markersize=self.markersize)
        plt.title(r'$\sin(\cdot)$' + ', ' + self.modeltype1)
        plt.xlabel('Size of the training sample')
        plt.legend(loc=2, prop={'size': 7}, ncol=2)
        fig = plt.gcf()
        fig.set_size_inches(3.75, 3)
        plt.savefig(self.folder_figure + '/thm3_R2={}.png'.format(str(R_2).replace('.', 'd')),
                    bbox_inches='tight', pad_inches=0.1, dpi=250)
