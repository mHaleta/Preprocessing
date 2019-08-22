import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import SelectKBest

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
    return res_set


#%%

def select_for_log(df):
    features = ['AvgCNPAmtClient',
                 'CNP_Payments_beforeClient',
                 'AvgCNPAmtTODClient',
                 'Merch_fraudlaent_cards_share',
                 'Merch_fraudlaent_transactions_share',
                 'avgCNPTOD_to_avg_CNP',
                 'Trans_to_Total_Amt',
                 'IncorrectTransactionsTod',
                 'CodesSentTOD'
                 ,'This_CCY_TOD_to_before'
                 ]
    
    return df[features].values

def select_not_log(df):
    features = ['Age',
                 'cards',
                 'Merch_transactions']
    return df[features].values    

def select_bin(df):
    features = ['Incorrect_transaction'
                ,'isRUR','isEUR','isUSD',
                 'c2c',
                 'epos_purchase',
                 'pre_auth'
                 ]
    return df[features].values    

#%%
    
def timestamp_to_date(ts):
    return "datetimefromparts({0},{1},{2},{3},{4},{5},0)".\
            format(ts.year,ts.month,ts.day,ts.hour,ts.minute,ts.second)

#%%
       
def decscribe_df_column(df, col_name, iqr_coeff=1.5, max_outliers_share=0.05):
    dtype = str(df[col_name].dtype)
    is_numeric = 1 if 'int' in dtype or 'float' in dtype else 0
    is_binary = 1 if df[col_name].drop_duplicates().shape[0]==2 else 0 
    is_date = 1 if 'date' in dtype else 0 
    is_categorical = 1 if dtype == 'object' else 0
    unique_values = df[col_name].drop_duplicates().shape[0]
    non_null_values = df[col_name].dropna().shape[0]
    total_values = df[col_name].shape[0]
    median = df[col_name].median() if is_numeric \
                else str(df[col_name].mode().tolist()[0] if df[col_name].mode().shape[0] else '')
    q25 = df[col_name].quantile(0.25) if is_numeric else None
    q75 = df[col_name].quantile(0.75) if is_numeric else None
    iqr = q75-q25 if is_numeric else None
    uppr_bound = q75+iqr_coeff*iqr if is_numeric else None
    lwr_bound = q25-iqr_coeff*iqr if is_numeric else None
    outliers_upper = (df[col_name]>uppr_bound).sum() if is_numeric else None
    outliers_lower = (df[col_name]<lwr_bound).sum() if is_numeric else None
    outliers_total = outliers_upper+outliers_lower if is_numeric else None
    is_for_log = 1 if is_numeric and not is_binary and outliers_total and non_null_values and outliers_total/non_null_values>=max_outliers_share else 0 
    return {'col_name':col_name,
            'dtype':dtype,
            'is_numeric':is_numeric,
            'is_binary':is_binary,
            'is_date':is_date,
            'is_categorical':is_categorical,
            'unique_values':unique_values,
            'non_null_values':non_null_values,
            'total_values':total_values,
            'median':median,
            'q25':q25,
            'q75':q75,
            'uppr_bound':uppr_bound,
            'lwr_bound':lwr_bound,
            'outliers_upper':outliers_upper,
            'outliers_lower':outliers_lower,
            'outliers_total':outliers_total,
            'is_for_log':is_for_log
            }
        
def describe_df(df,iqr_coeff=1.5):
    cols = df.columns.tolist()
    RESULT_COLS = ['col_name','dtype','is_numeric','is_binary','is_date'
                   ,'is_categorical','is_for_log','unique_values','non_null_values'
                   ,'total_values','median','q25','q75','uppr_bound'
                   ,'lwr_bound','outliers_upper','outliers_lower'
                   ,'outliers_total'] 
    res = []
    for col_name in cols:
        res.append(decscribe_df_column(df, col_name, iqr_coeff))
    return pd.DataFrame(res)[RESULT_COLS]
    
    
    
    
    
    