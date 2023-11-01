# imports
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

from matplotlib.ticker import PercentFormatter, FormatStrFormatter

from scipy import stats 
from sklearn.metrics import mutual_info_score

# load data preparation module
import src.data_prep as dp 

# create colors
c1 = sns.color_palette('Accent')[0]
c2 = sns.color_palette('Accent')[1]

# set alpha value for stat tests, 0.01 for 99% confidence level
alpha = 0.01 

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

def diabetes_piechart_ytrain(y_train):
    '''
    Plot the pie chart with diabetes % in the data
    '''
    # create values and labels for the pie chart
    values = pd.Series(y_train).value_counts().to_list()
    diabetes_labels = pd.Series(y_train).value_counts().index.to_list()
    diabetes_labels = ["Diabetes" if x==1 else "No diabetes" for x in diabetes_labels]
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

def sick_days_viz_bins(healthy: pd.DataFrame, diabetes: pd.DataFrame, bins: int = 3):
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
    axes[0].hist(healthy.MentHlth, bins=bins, color=c1, alpha=0.9, ec='black', label='Healthy', weights=np.ones(lh) / lh)
    axes[0].hist(diabetes.MentHlth, bins=bins, color=c2, alpha=0.9, ec = 'black', label = 'Diabetes', weights=np.ones(ld) / ld)
    axes[0].legend(loc='upper right', fontsize=15)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title('Physical', fontsize = 15)
    axes[1].hist(healthy.PhysHlth, bins=bins, color=c1, alpha=0.9, ec='black', label='Healthy', weights=np.ones(lh) / lh)
    axes[1].hist(diabetes.PhysHlth, bins=bins, color=c2, alpha=0.9, ec = 'black', label = 'Diabetes', weights=np.ones(ld) / ld)
    axes[1].legend(loc='upper right', fontsize=15)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

def age_viz(healthy: pd.DataFrame, diabetes: pd.DataFrame):
    ''' 

    '''
    h = pd.Categorical(healthy.Age, ordered=True)
    d = pd.Categorical(diabetes.Age, ordered=True)
    plt.figure(figsize=(12, 4))
    ax = sns.histplot(x=h, stat='percent', color=c1, label="Healthy", alpha = 0.5)
    ax = sns.histplot(x=d, stat='percent',  color=c2, label="Diabetes", alpha=0.5)
    #ax.set_xticklabels(rotation=30)
    plt.legend()
    plt.xticks(rotation=30, ha='right')
    #ax.tick_params(axis='x', rotation=30)
    plt.title('Age distribution')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
    plt.show()

############### STATISTICS ###############


def get_p_values(df: pd.DataFrame, cat_vars: list[str], target:str = 'Diabetes_binary', alpha:float = alpha):
    '''
    Performs Chi-squared test on categorical features of the dataframe.
    Parameters:
    df: data frame
    cat_vars: list of strings with names of categorical columns
    alpha: float, alpha value, default 0.1 for confidence interval 99%
    Returns a data frame with p_values of all categorical variables and their significance result
    '''

    #dictionary to hold names of the column and a p_value assotiated with it
    p_v = {}

    #for every column in category variables run a chi2 test
    for col in cat_vars:
        #create a crosstable
        observed = pd.crosstab(df[col], df[target])
        #run a chi squared test fot categorical data
        test = stats.chi2_contingency(observed)
        p_value = test[1].round(3)
        #add the result to the dictionary
        p_v[col] = p_value.round(3)
        
        #transform a dictionary to Series and then to Data Frame
        p_values = pd.Series(p_v).reset_index()
        p_values.rename(columns = {'index':'Feature', 0:'P_value'}, inplace = True)
        p_values = p_values.sort_values(by='P_value')

        #add the column that shows if the result is significant
        p_values['is_significant'] = p_values['P_value'] < alpha
    
    return p_values

def stat_categorical(df, target):
    ''' 

    '''

    # pull categorical values both nominal and ordinal
    ordinal = dp.ordinal
    nominal = dp.nominal
    categorical = nominal + ordinal

    # calculate mutual info scores and p_values
    mi_score = df[categorical].apply(lambda x: mutual_info_score(x, df[target]))
    p_values = get_p_values(df, categorical, target).set_index('Feature')

    return pd.concat([mi_score, p_values], axis = 1).rename({0:'mi'}, axis=1).sort_values(
        by='mi', ascending=False)

