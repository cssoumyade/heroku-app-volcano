import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error as mae
from model import *

def predict_time_to_erupt(seg_df):
    
    seg_df = seg_df.fillna(0)
    
    each_row = []
        
    for each_column in seg_df.columns:
        each_row.append(seg_df[each_column].std())
        each_row.append(seg_df[each_column].min())
        each_row.append(seg_df[each_column].max())
        each_row.append(seg_df[each_column].quantile(.3))
        each_row.append(seg_df[each_column].quantile(.6))
        each_row.append(seg_df[each_column].quantile(.8))
        each_row.append(seg_df[each_column].quantile(.9))
        each_row.append(seg_df[each_column].kurt())
    
    features = np.array(each_row).reshape(1,-1)
    
    with open('custEnsemblexgb.pkl', 'rb') as f:
        best_estimator = pickle.load(f)
        
    preds = best_estimator.predict(features)
    
    return preds[0]



def return_mae(seg_df, y):
    
    seg_df = seg_df.fillna(0)
    
    each_row = []
        
    for each_column in seg_df.columns:
        each_row.append(seg_df[each_column].std())
        each_row.append(seg_df[each_column].min())
        each_row.append(seg_df[each_column].max())
        each_row.append(seg_df[each_column].quantile(.3))
        each_row.append(seg_df[each_column].quantile(.6))
        each_row.append(seg_df[each_column].quantile(.8))
        each_row.append(seg_df[each_column].quantile(.9))
        each_row.append(seg_df[each_column].kurt())
    
    features = np.array(each_row).reshape(1,-1)
    
    with open('custEnsemblexgb.pkl', 'rb') as f:
        best_estimator = pickle.load(f)
        
    preds = best_estimator.predict(features)
    
    
    return mae(preds[0], y)    