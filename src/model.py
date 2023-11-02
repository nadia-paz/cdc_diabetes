import numpy as np 
import pandas as pd

import src.data_prep as dp

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

    
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

def train_model(df):
    # get train and y_train
    train, _, y_train, _ = dp.split_data(df, balance=True, full_train=True)
    # columns 
    binary = ['HighBP', 'HighChol', 'CholCheck',  'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'DiffWalk']
    ordinal_cat = ['MentHlth', 'PhysHlth']
    ordinal_num = ['GenHlth', 'Age', 'Education', 'Income']
    bmi = ["BMI_under", "BMI_over"]
    # change BMI
    train["BMI_under"] = train.BMI.map(lambda x: 1 if x <= 18 else 0)
    train = train.assign(BMI_over = train.BMI.apply(lambda x: 1 if x >= 25 else 0))

    # drop the column BMI
    train.drop("BMI", axis=1, inplace=True)

    # create One Hot Encoder
    ohe = OneHotEncoder(handle_unknown='error', drop='first', sparse=False)
    X = np.concatenate([
        train[binary],
        ohe.fit_transform(train[ordinal_cat + ordinal_num]).astype('uint8'),
        train[bmi].astype('uint8')
    ], axis=1)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth = 10,
        min_samples_leaf = 10,
        n_jobs=-1, # speed up the process
        random_state=dp.seed
        )
    rf.fit(X, y_train)
    return rf, ohe
