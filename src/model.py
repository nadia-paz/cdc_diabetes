import numpy as np 
import pandas as pd


import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

    
def get_report(y_val, y_pred, threshold=0.5):
    ''' 
    get classification report, remove unneeded information, add auc score
    Return:
        string to pass to print statement
    '''
    # get classification report
    report = classification_report(y_val, y_pred >= threshold).split("\n")
    # delete lines 2 (report for prediction of zeros) and 4 (empty string)
    del report[2]
    del report[3]

    # remove macro avg and weighted avg
    report = report[:-3]
    # insert empty string
    report.insert(3, "")
    # remove support column
    for i in range(len(report)):
        if len(report[i]) != 0:
            report[i] = report[i][:-10]
    # add roc_auc_score
    report.append(f"         auc                           {roc_auc_score(y_val, y_pred).round(2)}")

    return "\n".join(report)


def parse_xgb_output(output, df=True):
    ''' 
    Capture and parse XGBoost output, return result as a data frame.
    Important: use %%capture output magic command in Jupyter cell
    '''
    iterations = []
    train_scores = []
    validations_scores = []

    results = []

    # access every line of the code
    for line in output.stdout.strip().split('\n'):
        # split the line by tabulation
        num_iterations, train_score, val_score = line.split('\t')
        # parse strings and save numeric values
        it = int(num_iterations.strip('[]'))
        ts = float(train_score.split(':')[1])
        vs = float(val_score.split(':')[1])
        # append values to the lists
        iterations.append(it)
        train_scores.append(ts)
        validations_scores.append(vs)
        results.append((it, ts, vs, ts - vs))
        columns = ['iteration', 'train_score', 'validation_score', "score_diff"]
    if df:
        return pd.DataFrame(results, columns=columns)
    else:
        return (iterations, train_scores, validations_scores)
    
