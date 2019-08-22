import numpy as np
from sklearn.metrics import auc
import pandas as pd


#%%

def prepare_data_for_cap(y_true
                   , y_score):
    tmp = np.zeros((y_true.shape[0],6))
    tmp[:,0] = y_true
    tmp[:,1] = y_score
    tmp = tmp[(tmp[:,1]*-1).argsort()]
    tmp[:,2] = np.arange(1,y_true.shape[0]+1)
    tmp[:,3] = tmp[:,0].cumsum()
    rand_chance = y_true.mean()
    max_bad = y_true.sum()
    tmp[:,4] = rand_chance*tmp[:,2].copy()
    tmp[tmp[:,2]>max_bad,2] =  max_bad
    tmp[:,5] = np.arange(1,y_true.shape[0]+1)
    return tmp


def get_accuracy_score(for_cap_array):
    auc_actual = auc(for_cap_array[:,5],for_cap_array[:,3])
    auc_ideal = auc(for_cap_array[:,5],for_cap_array[:,2])
    auc_random = auc(for_cap_array[:,5],for_cap_array[:,4])
    return (auc_actual-auc_random)/(auc_ideal-auc_random)

def prepare_data_for_threshold_analysis(y_true
                   , y_score):
    thresh = np.round(np.arange(0,y_score.max(),0.01),2)
    analysis_array = np.zeros((thresh.shape[0],6))
    analysis_array[:,0] = thresh
    total_true = y_true.sum()
    for i,th in enumerate(thresh):
        total = (y_score>=th).sum()
        positives = y_true[y_score>=th].sum()
        precision = positives/total
        recall = positives/total_true
        f1 = 2*precision*recall/(precision+recall)
        analysis_array[i,1] = total
        analysis_array[i,2] = positives
        analysis_array[i,3] = precision
        analysis_array[i,4] = recall
        analysis_array[i,5] = f1
    colnames = ['Threshold','Total','Positives','Precision','Recall','F1']
    return pd.DataFrame(analysis_array,columns=colnames)


