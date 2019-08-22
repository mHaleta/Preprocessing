from .models_performance import prepare_data_for_cap, get_accuracy_score, prepare_data_for_threshold_analysis
from .plotting import plot_single_model_roc, plot_model_cap, plot_prec_recall_f1
import xlsxwriter
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import numpy as np

#%%

def create_classification_report(y_true,y_score,report_path,report_name):
    plots_path = os.path.join(report_path,report_name,'plots')
    excel_path = os.path.join(report_path,report_name,report_name+'.xlsx' if not report_name.endswith('.xlsx') else '')
    
    cap_data = prepare_data_for_cap(y_true,y_score)
    accuracy_score = get_accuracy_score(cap_data)
    
    thresh_analysis_data = prepare_data_for_threshold_analysis(y_true,y_score)
    
    plot_single_model_roc(y_true
                          ,y_score
                          ,save=True
                          ,savepath = plots_path
                          , file_name = 'roc_plot.png')
    
    plot_model_cap(cap_data
                   ,report_name
                   ,save=True
                   ,savepath = plots_path
                   , file_name = 'cap_plot.png')
    
    plot_prec_recall_f1(thresh_analysis_data
                   ,save=True
                   ,savepath = plots_path
                   , file_name = 'f1_plot.png')
    
    roc_auc = roc_auc_score(y_true=y_true,y_score=y_score)
    gini = 2*roc_auc-1
    f1 = thresh_analysis_data.F1.max()

    roc_auc = np.round(roc_auc,2)
    gini = np.round(gini,2)
    f1 = np.round(f1,2)
    accuracy_score = np.round(accuracy_score,2)
    
       
    print(excel_path)
    workbook = xlsxwriter.workbook.Workbook(excel_path,{'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet("Results")
    worksheet.set_column('A:A', 20)
    worksheet.write('A1', 'Metrics values:')
    worksheet.write('A2', 'ROC')
    worksheet.write('B2', roc_auc)
    worksheet.write('A3', 'GINI')
    worksheet.write('B3', gini)
    worksheet.write('A4', 'F1')
    worksheet.write('B4', f1)
    worksheet.write('A5', 'Accuracy Score')
    worksheet.write('B5', accuracy_score)
    worksheet.write('A7', 'Plots')
    
    worksheet.insert_image('A8', os.path.join(plots_path,'roc_plot.png'))
    worksheet.insert_image('A48', os.path.join(plots_path,'cap_plot.png'))
    worksheet.insert_image('A88', os.path.join(plots_path,'f1_plot.png'))
    
    fields = thresh_analysis_data.columns
    
    worksheet = workbook.add_worksheet("Table")
    worksheet.write('A1', fields[0])
    worksheet.write('B1', fields[1])
    worksheet.write('C1', fields[2])
    worksheet.write('D1', fields[3])
    worksheet.write('E1', fields[4])
    worksheet.write('F1', fields[5])
    
    index = 2
    for i, row in thresh_analysis_data.iterrows():

        threshold = row['Threshold']
        total = row['Total']
        positives = row['Positives']
        precision = row['Precision']
        recall = row['Recall']
        f1 = row['F1']
        
        worksheet.write('A{0}'.format(index+i),threshold)
        worksheet.write('B{0}'.format(index+i),total)
        worksheet.write('C{0}'.format(index+i),positives)
        worksheet.write('D{0}'.format(index+i),precision)
        worksheet.write('E{0}'.format(index+i),recall)
        worksheet.write('F{0}'.format(index+i),f1)
    
    workbook.close()
    
    
    