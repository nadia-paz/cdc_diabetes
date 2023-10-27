# imports
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

from scipy import stats 
from sklearn.metrics import mutual_info_score

# load data preparation module
import src.data_prep as dp 

# create colors
c1 = sns.color_palette('Accent')[0]
c2 = sns.color_palette('Accent')[1]

def autopct_format(values):
    '''
    the function accept value_counts from outcome_type
    puts it in % format ready to use in pie charts
    '''
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%)'.format(pct, v=val)
    return my_format

def diabetes_piechart(df: pd.DataFrame, target: str):
    '''
    Plot the pie chart with diabetes % in the data
    '''
    # create values and labels for the pie chart
    values = df[target].value_counts().to_list()
    diabetes_labels = df[target].value_counts().index.to_list()
    # create the pie chart
    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=diabetes_labels, explode=[0.01, 0.02], 
            colors=[c1, c2], autopct=autopct_format(values),
            shadow=False)
    plt.title('Diabetes among respondents')
    plt.show()


