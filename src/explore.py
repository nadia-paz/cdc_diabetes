# imports
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

from matplotlib.ticker import PercentFormatter

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
        return '{:.1f}%'.format(pct, v=val)
    return my_format

def diabetes_piechart(df: pd.DataFrame, target: str):
    '''
    Plot the pie chart with diabetes % in the data
    '''
    # create values and labels for the pie chart
    values = df[target].value_counts().to_list()
    diabetes_labels = df[target].value_counts().index.to_list()
    # create the pie chart
    plt.figure(figsize=(4, 4))
    plt.pie(values, labels=diabetes_labels, explode=[0.01, 0.02], 
            colors=[c1, c2], autopct=autopct_format(values),
            shadow=False)
    plt.title('Diabetes among respondents')
    plt.show()

def bmi_distribution(healthy: pd.DataFrame, diabetes: pd.DataFrame):
    ''' 
    Plot 2 histograms for BMI distribution among those who has diabetes and don't
    '''
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    axes[0].hist(healthy.BMI, color = c1, alpha = 0.7, edgecolor='black')
    axes[0].set_title('BMI distribution for healthy people')
    axes[1].hist(diabetes.BMI, color = c2, alpha = 0.7, edgecolor='black')
    axes[1].set_title('BMI distribution for people with diabetes')
    plt.show()

def bmi_piechart(under: pd.DataFrame, normal: pd.DataFrame, over: pd.DataFrame):
    ''' 
    Creates 3 pie charts with weight distribution for ppl with underweight, normal weight and overweith
    '''
    plt.figure(figsize = (12, 4))
    target = 'Diabetes'
    for i, j in enumerate(zip([under, normal, over], ['low BMI', 'normal BMI', 'high BMI'])):
        plt.subplot(1, 3, i+1)
        values = j[0][target].value_counts().to_list()
        diabetes_labels = j[0][target].value_counts().index.to_list()
        plt.pie(values, labels=diabetes_labels, explode=[0.01, 0.02], 
            colors=[c1, c2], autopct=autopct_format(values))
        plt.title(j[1])
    plt.show()

def sick_days_viz(healthy: pd.DataFrame, diabetes: pd.DataFrame):
    ''' 
    Vizualize the number of poor menthal and physical health among respondents with and without diabetes 
    '''
    # length of data sets. we need it to convert values in histograms from absolute to relative
    lh, ld = len(healthy), len(diabetes)

    #plt.figure(figsize = (12, 4))
    # left plot -> mental health
    #plt.subplot(121)
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    plt.suptitle("Days of poor health per month", fontsize = 15)
    axes[0].set_title('Mental', fontsize = 15)
    axes[0].hist(healthy.MentHlth, color=c1, alpha=0.9, ec='black', label='Healthy', weights=np.ones(lh) / lh)
    axes[0].hist(diabetes.MentHlth, color=c2, alpha=0.9, ec = 'black', label = 'Diabetes', weights=np.ones(ld) / ld)
    axes[0].legend(loc='upper right', fontsize=15)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title('Physical', fontsize = 15)
    axes[1].hist(healthy.PhysHlth, color=c1, alpha=0.9, ec='black', label='Healthy', weights=np.ones(lh) / lh)
    axes[1].hist(diabetes.PhysHlth, color=c2, alpha=0.9, ec = 'black', label = 'Diabetes', weights=np.ones(ld) / ld)
    axes[1].legend(loc='upper right', fontsize=15)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()


def calculate_mi(series):
    '''
    check the mutual info score
    '''
    return mutual_info_score(series, df_explore.Diabetes_binary)