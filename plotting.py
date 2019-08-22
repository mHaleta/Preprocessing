from matplotlib import pyplot as plt
import seaborn as sns
import os
from .stats import get_boot_stats
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np

#%%

def plot_violin(x,y,title,save=False,xname=None,yname=None,savepath='',file_name=''):
    if not file_name:
        file_name = title+'.png'
    plt.figure(figsize=(8,8))
    sns.violinplot(y=y,x=x)
    plt.title(title,fontsize=15)
    if xname:
        plt.xlabel(xname,fontsize=15)
    if yname:
        plt.ylabel(yname,fontsize=15)
    if save:
        if savepath:
            os.makedirs(savepath,exist_ok=True)
            path = os.path.join(savepath,file_name)
        else:
            path = file_name
        print(path)
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
    
def plot_boot(x,y
              ,statfunc
              ,iterations=1000
              ,iterative=False
              ,xname=''
              ,yname=''
              ,title=''
              ,cmap=None
              ,save=False
              ,savepath=''
              ,file_name=''
              ):
    if not file_name:
        file_name = title+'.png'
    y_classes = np.unique(y)
    boots = []
    legends = []
    colors = []
    cols_to_choose = np.arange(0,255)
    np.random.shuffle(cols_to_choose)
    cols_to_choose = cols_to_choose.tolist()
    for i,c in enumerate(y_classes):
        if not cmap:
            col = plt.cm.rainbow(cols_to_choose.pop())
        else:
            col = cmap[c]
        boots.append(get_boot_stats(x[y==c]
                    ,statfunc=statfunc
                    ,iterations=iterations
                    ,iterative=iterative))
        colors.append(col)
        legends.append(yname+'='+str(c))
    plt.figure(figsize=(8,8))
    for i,boot in enumerate(boots):
        sns.distplot(boot,label=legends[i],color=colors[i])
    plt.legend(loc=2)
    plt.title(title,fontsize=15)
    plt.xlabel(yname,fontsize=15)
    if save:
        if savepath:
            os.makedirs(savepath,exist_ok=True)
            path = os.path.join(savepath,file_name)
        else:
            path = file_name
        print(path)
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
    
def plot_model_roc(y_true_train
                   , y_score_train
                   , y_true_test
                   , y_score_test
                   , model_name=None
                   , save = False
                   , savepath = ''
                   , file_name = ''):
    
    fpr_train,tpr_train,_ = roc_curve(y_score=y_score_train,y_true=y_true_train)
    fpr_test,tpr_test,_ = roc_curve(y_score=y_score_test,y_true=y_true_test)
    score_train = np.round(roc_auc_score(y_score=y_score_train,y_true=y_true_train),2)
    score_test = np.round(roc_auc_score(y_score=y_score_test,y_true=y_true_test),2)
    
    plt.figure(figsize=(8,8))
    plt.plot(fpr_train,tpr_train,color='brown',label='train (AUC={0})'.format(score_train))
    plt.plot(fpr_test,tpr_test,color='orange',label='test (AUC={0})'.format(score_test))
    plt.plot(fpr_test,fpr_test,color='darkgrey',linestyle='--',linewidth=0.75)
    plt.title("ROC{0}".format(" (model = "+model_name+")" if model_name else ''),
              fontsize=15)
    plt.xlabel("FPR",fontsize=15)
    plt.ylabel("TPR",fontsize=15)
    plt.legend(loc=4)
    
    if save:
        if savepath:
            os.makedirs(savepath,exist_ok=True)
            path = os.path.join(savepath,file_name)
        else:
            path = file_name
        print(path)
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def plot_single_model_roc(y_true
                   , y_score
                   , model_name=None
                   , save = False
                   , savepath = ''
                   , file_name = ''):
    
    fpr_train,tpr_train,_ = roc_curve(y_score=y_score,y_true=y_true)
    score_train = np.round(roc_auc_score(y_score=y_score,y_true=y_true),2)
    
    plt.figure(figsize=(8,8))
    plt.plot(fpr_train,tpr_train,color='brown',label='Model (AUC={0})'.format(score_train))
    plt.plot(tpr_train,tpr_train,color='darkgrey',linestyle='--',linewidth=0.75)
    plt.title("ROC{0}".format(" (model = "+model_name+")" if model_name else ''),
              fontsize=15)
    plt.xlabel("FPR",fontsize=15)
    plt.ylabel("TPR",fontsize=15)
    plt.legend(loc=4)
    
    if save:
        if savepath:
            os.makedirs(savepath,exist_ok=True)
            path = os.path.join(savepath,file_name)
        else:
            path = file_name
        print(path)
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def plot_model_cap(for_cap_array
                   , model_name
                   , accuracy=None
                   , save = False
                   , savepath = ''
                   , file_name = ''):
    plt.figure(figsize=(8,8))
    plt.plot(for_cap_array[:,5]
            ,for_cap_array[:,3]
            ,color='brown'
            ,label="Actual model")
    plt.plot(for_cap_array[:,5]
            ,for_cap_array[:,4]
            ,color='darkgrey'
            ,linestyle='--'
            ,label="Random choice")
    plt.plot(for_cap_array[:,5]
            ,for_cap_array[:,2]
            ,color='red'
            ,label="Ideal choice")
    plt.legend(loc=4)   
    plt.xlabel("Total positives", fontsize=15)
    plt.ylabel("True positives", fontsize=15)
    title = "CAP curve for {0}".format(model_name)
    if accuracy is not None:
        title = title + ", accuracy score = {0}".format(accuracy)
    plt.title(title, fontsize=15)

    if save:
        if savepath:
            os.makedirs(savepath,exist_ok=True)
            path = os.path.join(savepath,file_name)
        else:
            path = file_name
        print(path)
        plt.savefig(path)
    else:
        plt.show()
    plt.close()    
    
def plot_prec_recall_f1(threshold_data
                        , save = False
                        , savepath = ''
                        , file_name = ''):
    plt.figure(figsize=(8,8))
    plt.plot(threshold_data.Threshold.values
             ,threshold_data.Precision.values
             ,color='orange'
             ,label='Precision')
    plt.plot(threshold_data.Threshold.values
             ,threshold_data.Recall.values
             ,color='green'
             ,label='Recall')
    plt.plot(threshold_data.Threshold.values
             ,threshold_data.F1.values
             ,color='red'
             ,label='F1')
    plt.legend(loc=2)
    f1 = threshold_data.F1.values
    thresh = threshold_data.Threshold.values
    f1_bool_list = f1==f1.max()
    f1_index = np.array(thresh[f1_bool_list])
    best_thresh_for_title = ','.join([str(t) for t in f1_index])

    plt.vlines(f1_index,0,1,color='darkgrey',linestyle='--',label='Best F1')
    plt.title("F1 plot\nbest F1 = {0}, threshold = {1}".format(np.round(f1.max(),2)
                                                                ,best_thresh_for_title)
                , fontsize=15)
    plt.xlabel("Threshold", fontsize=15)
    plt.ylabel("Measures Values", fontsize=15)
    
    if save:
        if savepath:
            os.makedirs(savepath,exist_ok=True)
            path = os.path.join(savepath,file_name)
        else:
            path = file_name
        print(path)
        plt.savefig(path)
    else:
        plt.show()
    plt.close()    
        