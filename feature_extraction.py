import numpy as np
from scipy.stats import mannwhitneyu
from .preprocessing import get_feature_set
from sklearn.preprocessing import QuantileTransformer

#%%

def generate_features_combinations(columns):
    colnum = len(columns)
    for i in range(colnum-1):
        for j in range(i+1,colnum):
            yield (columns[i],columns[j])

def divide_col_by_col(df,col1,col2,log=False):
    X1 = np.float64(df[col1].fillna(df[col1].median()).values)
    X2 = np.float64(df[col2].fillna(df[col2].median()).values)
    if log:
        X1 = np.log(np.abs(X1)+1)
        X2 = np.log(np.abs(X2)+1)
    X1[X1==0] = 0.1
    X2[X2==0] = 0.1
    return X1/X2

def get_new_features_by_div(df
                            ,y
                            ,columns
                            ,conf_level=0.05
                            ,min_diff=0.1
                            ,log=False
                            ,verbose=False
                            ,corr_lim=0.5
                            ,only_positive=True):
    combs = generate_features_combinations(columns)
    res = np.random.normal(size=(df.shape[0],2))
    colnames = []
    worth_to_check = []
    i = 0 
    last_extracted = 0
    for col1,col2 in combs:
        i+=1
        X = divide_col_by_col(df,col1,col2,log)
        p = mannwhitneyu(X[y==0],X[y==1]).pvalue
        diff = np.abs(np.median(X[y==0])-np.median(X[y==1]))/(np.median(X)+0.01)
        if p<= conf_level and diff>=min_diff:
            
            new_col_name = '{0}_DIV_{1}'.format(col1,col2)
            print(new_col_name,diff,np.median(X[y==0]),np.median(X[y==1]))
            res = np.hstack((res,X.reshape((-1,1))))
            colnames.append(new_col_name)
            if only_positive:
                selected_features = get_feature_set(QuantileTransformer().fit_transform(res)
                                                    ,y
                                                    ,corr_lim=corr_lim)
            else:
                selected_features = get_feature_set(res
                                                    ,y
                                                    ,corr_lim=corr_lim)
            res = res[:,selected_features]

            if res.shape[1]<3:
                res = res.hstack((res,np.random.normal(size=(df.shape[0],3-res.shape[1]))))
            colnames = np.array(colnames)[selected_features[2:]].tolist()
            if verbose and new_col_name in colnames:
                print('Divided: ',col1,col2)
        if verbose:
            print(i,' combinations processed ',res.shape[1]-2,' features extracted')
        if selected_features[-1]:
            worth_to_check.append((col1,col2))
            print('='*20,'WORTH TO CHECK','='*20)
            print((col1,col2))
            print('Total worth to check:', len(worth_to_check))
            print('='*54)
        last_extracted = res.shape[1]-2
    if res.shape[1]==1:
        return None,None
    return res[:,1:], colnames, worth_to_check

#%%



#%%

a = np.random.randint(0,100,(10,3))

#%%

a[:,[True,False,True]]
