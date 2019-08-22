import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#%%

class LogTransformer(BaseEstimator,TransformerMixin):

    def fit(self,X,y=None):
        return self
        
    def transform(self,X,y=None):
        return np.log(np.abs(X)+1)
    
    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)

class OutliersCutter(BaseEstimator,TransformerMixin):
    
    def __init__(self, n=1.5):
        self.upper = 0
        self.lower = 0
        self.n = n
    
    def fit(self,X,y=None):
        q1 = np.percentile(X,25,axis=0)
        q2 = np.percentile(X,75,axis=0)
        iqr = q2-q1
        self.upper = q2+self.n*iqr
        self.lower = q1-self.n*iqr
        return self
        
    def transform(self,X,y=None):
        for i in range(X.shape[1]):
            tmp_up = self.upper[i]
            tmp_low = self.lower[i]
            tmp = X.copy()
            for_repl = tmp[:,i] > tmp_up
            tmp[for_repl,i] = tmp_up
            for_repl = tmp[:,i] < tmp_low
            tmp[for_repl,i] = tmp_low
        return tmp
    
    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)
#%%
class CorrelationsRemover(BaseEstimator,TransformerMixin):
    
    def __init__(self,corr_lim=0.5,verbose=0):
        self.support = None
        self.corr_lim = corr_lim
        self.verbose = verbose
    
    def diffbool(self,a,b):
        tmp1 = np.int64(a)
        tmp2 = np.int64(b)
        res = tmp1-tmp2
        res[res<0]=0
        return np.array(res,np.bool)
    
    
    def get_excluded_cols(self,check_col_index,corrmat,corr_lim=0.5):
        exclude = np.abs(corrmat[check_col_index,:])>=corr_lim
        return exclude[0]
    
    def correct_selected_feature(self,cols_to_check,selected_feature):
        sel_feature_index = 0
        res = np.array([False for i in range(cols_to_check.shape[0])])
        for i in range(cols_to_check.shape[0]):
            if cols_to_check[i]:
                if selected_feature[sel_feature_index]==True:
                    res[i]=True
                    return res
                sel_feature_index += 1
        return res
                    
    
    def fit(self,X,y=None):
        corr_lim = self.corr_lim
        verbose = self.verbose
        res_set = np.array([False for i in range(X.shape[1])])
        cols_to_check = np.array([True for i in range(X.shape[1])])
        cor = spearmanr(X)
        i=0
        while cols_to_check.sum()>0:
            if verbose:
                print('iteration',i)
            skbest = SelectKBest(chi2,k=1).fit(X[:,cols_to_check],y)
            selected_feature = skbest.get_support()
            selected_feature = self.correct_selected_feature(cols_to_check,selected_feature)
            
            res_set = res_set + selected_feature
            to_exclude = self.get_excluded_cols(selected_feature,cor[0],corr_lim)
            
            cols_to_check = self.diffbool(cols_to_check,to_exclude)
            if verbose:
                print('iteration',i)
                print('to exclude',to_exclude)
                print('selected feature',selected_feature)
                print('left to check',cols_to_check)
            i+=1
        if verbose:
            print('selected_set',res_set)
        self.support = res_set
        return self
    
    def transform(self,X,y=None):
        return X[:,self.support]
    
  #%%  
    

def get_outliers(X, n=3):
    q1 = np.percentile(X,25,axis=0)
    q2 = np.percentile(X,75,axis=0)
    iqr = q2-q1
    upper = q2+n*iqr
    lower = q1-n*iqr
    return upper, lower

def replace_out(X,upper,lower):
    for i in range(X.shape[1]):
        tmp_up = upper[i]
        tmp_low = lower[i]
        for_repl = X[:,i] > tmp_up
        X[for_repl,i] = tmp_up
        for_repl = X[:,i] < tmp_low
        X[for_repl,i] = tmp_low
      
     

def diffbool(a,b):
    tmp1 = np.int64(a)
    tmp2 = np.int64(b)
    res = tmp1-tmp2
    res[res<0]=0
    return np.array(res,np.bool)


def get_excluded_cols(check_col_index,corrmat,corr_lim=0.5):
    exclude = np.abs(corrmat[check_col_index,:])>=corr_lim
    return exclude[0]

def correct_selected_feature(cols_to_check,selected_feature):
    sel_feature_index = 0
    res = np.array([False for i in range(cols_to_check.shape[0])])
    for i in range(cols_to_check.shape[0]):
        if cols_to_check[i]:
            if selected_feature[sel_feature_index]==True:
                res[i]=True
                return res
            sel_feature_index += 1
    return res
                

def get_feature_set(X,y,corr_lim=0.5,verbose=0):
    res_set = np.array([False for i in range(X.shape[1])])
    cols_to_check = np.array([True for i in range(X.shape[1])])
    cor = spearmanr(X)
    i=0
    while cols_to_check.sum()>0:
        if verbose:
            print('iteration',i)
        skbest = SelectKBest(chi2,k=1).fit(X[:,cols_to_check],y)
        selected_feature = skbest.get_support()
        selected_feature = correct_selected_feature(cols_to_check,selected_feature)
        
        res_set = res_set + selected_feature
        to_exclude = get_excluded_cols(selected_feature,cor[0],corr_lim)
        
        cols_to_check = diffbool(cols_to_check,to_exclude)
        if verbose:
            print('iteration',i)
            print('to exclude',to_exclude)
            print('selected feature',selected_feature)
            print('left to check',cols_to_check)
        i+=1
    if verbose:
        print('selected_set',res_set)
    return res_set

#%%
    
class DFTransformer(BaseEstimator,TransformerMixin):
    
    def __init__(self
                    ,y_name
                    ,not_log=None
                    ,for_log=None
                    ,binary=None
                    ,categorical=None
                    ,outliers_limit=1.5
                    ,cut_outliers=True
                    ,use_corr_remover=True
                    ,corr_lim=0.5
                    ,use_variance_threshold=True
                    ,verbose=0):
        self.imputers = {'log':SimpleImputer(strategy='median'),
                         'not_log':SimpleImputer(strategy='median'),
                         'bin':SimpleImputer(strategy='most_frequent'),
                         'cat':SimpleImputer(strategy='constant',fill_value='NA')}
        
        self.not_log = not_log
        self.for_log = for_log
        self.binary = binary
        self.categorical = categorical
        self.y_name = y_name
        
        self.cut_outliers = cut_outliers
        self.use_corr_remover = use_corr_remover
        
        self.use_variance_threshold = use_variance_threshold
        self.log_transformer = LogTransformer()
        self.out_cutter = OutliersCutter(n=outliers_limit)
        self.corr_remover = CorrelationsRemover(corr_lim=corr_lim)
        self.vt = VarianceThreshold(0.0001)
        self.NOT_LOG_PRESENT = not_log is not None and len(not_log)>0
        self.LOG_PRESENT = for_log is not None and len(for_log)>0
        self.BINARY_PRESENT = binary is not None and len(binary)>0
        self.CAT_PRESENT = categorical is not None and len(categorical)>0
        self.verbose = verbose
        f1 = self.not_log if self.NOT_LOG_PRESENT else []
        f2 = self.for_log if self.LOG_PRESENT else []
        self.num_features = f1 + f2 
        
#        if verbose:
#            print('Num Features:',self.num_features)
#            print(self.LOG_PRESENT,self.for_log)
        self.NUMERIC_PRESENT = self.num_features is not None and len(self.num_features)>0
        self.final_features = None
        
    def fit(self,X,y=None):

        if self.NOT_LOG_PRESENT:
            if self.verbose:
                print('Processing numeric features')
                print('not_log include',len(self.not_log), 'values')
            not_log_df = X[self.not_log].copy()
            self.imputers['not_log'].fit(not_log_df.values)
            not_log_df = self.imputers['not_log'].transform(not_log_df.values) 
        else:
            not_log_df = None
                
        
        if self.LOG_PRESENT:
            if self.verbose:
                print('Processing log features')
                print('for_log include',len(self.for_log), 'values')
            for_log_df = X[self.for_log].copy()
            self.imputers['log'].fit(for_log_df.values)
            for_log_df = self.imputers['log'].transform(for_log_df.values)
            for_log_df = self.log_transformer.transform(for_log_df)
        else:
            for_log_df = None
        
        if self.BINARY_PRESENT:
            if self.verbose:
                print('Processing binary features')
            binary_df = X[self.binary].copy()
            self.imputers['bin'].fit(binary_df.values)
            binary_df = self.imputers['bin'].transform(binary_df.values)
        else:
            binary_df = None
            
        if self.CAT_PRESENT:
            if self.verbose:
                print('Processing categorical features')
            categorical_df = X[self.categorical].copy()
            self.imputers['cat'].fit(categorical_df.values)
            categorical_df = self.imputers['cat'].transform(categorical_df.values)
        else:
            categorical_df = None
            
        if self.NUMERIC_PRESENT:
            if self.verbose:
                print('Joining numeric features')
                print('\tnot_log include',len(self.not_log), 'values')
                print('\tfor_log include',len(self.for_log), 'values')
                print('\ttotal num features',len(self.num_features))
            if self.NOT_LOG_PRESENT and self.LOG_PRESENT:
                num_df = np.hstack((not_log_df,for_log_df))
            else:
                num_df = not_log_df if self.NOT_LOG_PRESENT else for_log_df 
        else:
            num_df = None
            
        if self.NUMERIC_PRESENT and self.cut_outliers:
            if self.verbose:
                print('Cutting outliers')
                print('\tnot_log include',len(self.not_log), 'values')
                print('\tfor_log include',len(self.for_log), 'values')
                print('\ttotal num features',len(self.num_features))
            self.out_cutter.fit(num_df)
            num_df = self.out_cutter.transform(num_df)
            
        if self.NUMERIC_PRESENT and self.use_corr_remover:
            if self.verbose:
                print('Removing correlations')
                print('\tnot_log include',len(self.not_log), 'values')
                print('\tfor_log include',len(self.for_log), 'values')
                print('\ttotal num features',len(self.num_features))
            mms = MinMaxScaler().fit(num_df)
            self.corr_remover.fit(mms.transform(num_df),y)
            self.num_features = np.array(self.num_features)[self.corr_remover.support].tolist()
            num_df = self.corr_remover.transform(num_df)
            
        if self.NUMERIC_PRESENT and self.use_variance_threshold:
            if self.verbose:
                print('Removing low variance')
            self.vt.fit(num_df)
            self.num_features = np.array(self.num_features)[self.vt.get_support()].tolist()
            
            if self.verbose:
                    print(self.vt.get_support().shape[0]-self.vt.get_support().sum(),' features removed!')
                    print('\tnot_log include',len(self.not_log), 'values')
                    print('\tfor_log include',len(self.for_log), 'values')
                    print('\ttotal num features',len(self.num_features))
#        res_df = [d for d in [num_df,binary_df,categorical_df] if d is not None]
            
        self.final_features = self.num_features\
                        +(self.binary if self.BINARY_PRESENT else [])\
                        +(self.categorical if self.CAT_PRESENT else [])
#        print(self.final_features)
        
    def transform(self,X,y=None):
        
        if self.NOT_LOG_PRESENT:
            if self.verbose:
                print('Processing numeric features')
            not_log_df = X[self.not_log].copy()
            not_log_df = self.imputers['not_log'].transform(not_log_df.values) 
        else:
            not_log_df = None
                
        
        if self.LOG_PRESENT:
            if self.verbose:
                print('Processing log features')
            for_log_df = X[self.for_log].copy()
            for_log_df = self.imputers['log'].transform(for_log_df.values)
            for_log_df = self.log_transformer.transform(for_log_df)
        else:
            for_log_df = None
        
        if self.BINARY_PRESENT:
            if self.verbose:
                print('Processing binary features')
            binary_df = X[self.binary].copy()
            binary_df = self.imputers['bin'].transform(binary_df.values)
        else:
            binary_df = None
            
        if self.CAT_PRESENT:
            if self.verbose:
                print('Processing categorical features')
            categorical_df = X[self.categorical].fillna('NA').copy().values

        else:
            categorical_df = None
            
        if self.NUMERIC_PRESENT:
            if self.NOT_LOG_PRESENT and self.LOG_PRESENT:
                num_df = np.hstack((not_log_df,for_log_df))
            else:
                num_df = not_log_df if self.NOT_LOG_PRESENT else for_log_df 
        else:
            num_df = None
            
        if self.NUMERIC_PRESENT and self.cut_outliers:
            num_df = self.out_cutter.transform(num_df)
            
        if self.NUMERIC_PRESENT and self.use_corr_remover:
            num_df = self.corr_remover.transform(num_df)
            
        if self.NUMERIC_PRESENT and self.use_variance_threshold:
            if self.verbose:
                print('Removing low variance')
            num_df = self.vt.transform(num_df)
            
#        print(self.final_features)
        res_df = [d for d in [num_df,binary_df,categorical_df] if d is not None]
        res_df = np.hstack(tuple(res_df))
        
        res_df = pd.DataFrame(res_df,columns=self.final_features)
        res_df[self.y_name] = X[self.y_name].values if y is None else y
        
        return res_df
        
        
        


#a = spearmanr(train[num_features].values[:,:3])[0]