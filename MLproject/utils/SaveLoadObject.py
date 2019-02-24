'''
Created on Feb 10, 2019

@author: mofir
'''
import pickle
import pandas as pd

def save_obj(obj, name ):
    with open(name+'.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name+'.pickle', 'rb') as f:
        return pickle.load(f)


def AddNewSheetXls(df_list,path,sheet_names_list):
    writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
    for name,df in zip(sheet_names_list,df_list):
        df.to_excel(writer, sheet_name = name[0])
    writer.save()
    writer.close()