import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

seed = 2912
# categorical data
ordinal = ['GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

nominal = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
       'DiffWalk', 'Sex']

# columns for preprocessing and modeling
binary = ['HighBP', 'HighChol', 'CholCheck',  'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'DiffWalk']
ordinal_cat = ['MentHlth', 'PhysHlth']
ordinal_num = ['GenHlth', 'Age', 'Education', 'Income']
numerical = ['BMI']

def acquire():
    ''' 
    checks if the data folder and the diabetis_data file already exist.
    if yes - returns data
    if no - first creates a folder, than downloads data from the UC Irvine Machine Learning Repository,
    saves it as a csv file and returns 
    '''

    dir='data'
    filename='diabetis_data.csv'
    path_to_file = os.path.join(dir, filename)
    if os.path.isfile(path_to_file):
        df = pd.read_csv(path_to_file)
    else:
            # fetch dataset 
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
        
        # data (as pandas dataframes) 
        X = cdc_diabetes_health_indicators.data.features 
        y = cdc_diabetes_health_indicators.data.targets 
        # merge into a data frame
        df = pd.concat([X, y], axis = 1)
        # save file
        if not os.path.exists(dir):
            try:
                os.mkdir(dir)
            except OSError as e:
                print(e)
        try:
            df.to_csv(path_to_file, index_label=False)
        except OSError as e:
                print(e)
    # drop 24206 duplicates
    df = df.drop_duplicates()


    #df.Diabetes_binary = pd.Categorical(df.Diabetes_binary)

    # rename the target variable
    #df.rename({'Diabetes_binary':'Diabetes'}, axis=1, inplace=True)

    return df

def replace_values(full_train: pd.DataFrame):
    ''' 
    Prepares data for exploration by renaming values of categorical variables from numerical to meaningful words.
    Parameters:
        df: pandas data frame
    Returns:
        data frame with replaced values
    '''

    # create dictionaries for non-binary columns
    age_dict = {
        1: "18 - 24",
        2: "25 - 29",
        3: "30 - 34",
        4: "35 - 39",
        5: "40 - 44",
        6: "45 - 49",
        7: "50 - 54",
        8: "55 - 59",
        9: "60 - 64",
        10: "65 - 69",
        11: "70 - 74",
        12: "75 - 79",
        13: "80 and older"
    }

    edu_dict = {
        1: "no school",    #"no high school",
        2: "elementary school",     #"no high school",
        3: "middle school",     #"no high school",
        4: "high school",
        5: "some college",
        6: "college degree"
    }

    income_dict = {
        1: "less then $10K",
        2: "$10K - $15K",
        3: "$15K - $20K",
        4: "$20K - $25K",
        5: "$25K - $35K",
        6: "$35K - $50K",
        7: "$50K - $75K",
        8: "more than $85K"
    }
    df = full_train.copy()
    # replace values
    df.HighBP = np.where(df.HighBP == 0, "Normal Blood Pressure", 'High Blood Pressure')
    df.HighChol = np.where(df.HighChol == 0, "Normal Cholesterol", 'High Cholesterol')
    df.CholCheck = np.where(df.CholCheck == 0, "No Cholesterol Check", "Had Cholesterol Check")
    df.Smoker = np.where(df.Smoker == 0, 'Not a Smoker', 'Smoker')
    df.Stroke = np.where(df.Stroke == 0, "Didn't Have a Stroke", "Had a Stroke")
    df.HeartDiseaseorAttack = np.where(df.HeartDiseaseorAttack == 0, 'No Heart Disease', 'Heart Disese')
    df.PhysActivity = np.where(df.PhysActivity == 0, "No Physical Activity - 30 days", "Active in the past 30 days")
    df.Fruits = np.where(df.Fruits == 0, 'Doesn\'t eat fruits', 'Eats at least 1 friut a day')
    df.Veggies = np.where(df.Veggies == 0, 'Doesn\'t eat veggies', 'Eats at least 1 vegetable a day')
    df.HvyAlcoholConsump = np.where(df.HvyAlcoholConsump == 0, "Normal alcohol consumption", "Heavy alcohol consumption")
    df.AnyHealthcare = np.where(df.AnyHealthcare == 0, 'No medical insurance', 'Medical Insurance')
    df.NoDocbcCost = np.where(df.NoDocbcCost == 0, 'Visit doctor when sick', 'No doctor visit becasue of cost')
    df.DiffWalk = np.where(df.DiffWalk == 0, 'Normal walking', 'Difficulty walking')
    df.Sex = np.where(df.Sex == 0, 'Female', 'Male')
    df.Age = df.Age.replace(age_dict)
    df.Education = df.Education.replace(edu_dict)
    df.Income = df.Income.replace(income_dict)
    df["Diabetes"] = np.where(df.Diabetes_binary == 0, 'No Diabetes', 'Diabetes')

    return df

def clean_data(df: pd.DataFrame):
    ''' 
    Drop the column with low mutual info score.  
    '''
    df_new = df.copy()
    # drop columns
    df_new = df_new.drop(['Veggies', 'Sex', 'Fruits', 'AnyHealthcare', 'NoDocbcCost'], axis=1)
    # reduce number of ordered categories down to three, automatically saves as a category type
    df_new.MentHlth = pd.cut(df_new.MentHlth, bins=3, labels=['low', 'medium', 'high'])
    df_new.PhysHlth = pd.cut(df_new.PhysHlth, bins=3, labels=['low', 'medium', 'high'])

    # turn ordinar data into categories
    ord = ['GenHlth', 'Age', 'Education', 'Income']
    for col in ord:
        df_new[col] = pd.Categorical(df_new[col]).as_ordered()
    # turn binary data and BMI to uint8 (0 to 255)
    binary = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 
       'DiffWalk', 'Diabetes_binary']
    for col in binary + ['BMI']:
        df_new[col] = df_new[col].astype('uint8')

    return df_new

def balance_data(df, target='Diabetes_binary', replace=False):
    '''
    Balance data sets. Binary classification only. By deafault -> downsampling.
    Parameters:
        df: pandas data frame to balance
        replace: boolean. 
        If True the functions makes upsampling by performing replacement with repetitions (sampling with replacement)
        If False the function makes downsampling, no repetitions
    Returns:
        balance data frame with equal number of people with and without diabetes
    '''
    # get x and y. x - the value of majority, y - the value of minority
    x = df[target].value_counts().index[0]
    y = df[target].value_counts().index[1]
    # Separate majority and minority classes
    majority = df[df[target] == x] # majority
    minority = df[df[target] == y] # minority
    # for upscaling
    if replace:
        # Upscale the minority class
        sampled_data = resample(minority, 
                            replace=replace,     # sample with replacement
                            n_samples=len(majority),    # to match majority class
                            random_state=seed) # reproducible results
        df_sampled = pd.concat([majority, sampled_data])
    # for downscaling
    else:
        # Downsample majority class
        sampled_data = resample(majority, 
                            replace=False,    # sample without replacement
                            n_samples=len(minority), # to match minority class
                            random_state=seed) # reproducible results
    
        # Combine minority class with downsampled majority class
        df_sampled = pd.concat([sampled_data, minority])

    return df_sampled

def change_bmi(df: pd.DataFrame):
    ''' 
    1. Changes BMI column to 2 columns: 
    BMI_under -> if BMI <= 18 -> 1, else 0
    BMI_over -> if BMI >= 25 -> 1, else 0
    Normal BMI gets 0-0 from both columns created
    2. Drops the original BMI column.

    Returns: 
        new data frame with updated columns
    '''
    df_new = df.copy()

    # don't use np.where not to forget other ways :)
    df_new["BMI_under"] = df_new.BMI.map(lambda x: 1 if x <= 18 else 0)
    df_new = df_new.assign(BMI_over = df_new.BMI.apply(lambda x: 1 if x >= 25 else 0))

    # drop the column BMI
    df_new.drop("BMI", axis=1, inplace=True)

    return df_new

def split_data(df, explore=False, balance=False, replace=False, target = 'Diabetes_binary'):
    '''
    Split data for exploration and machine learning models.
    The function applies data manipulations according to the boolean parameters. 
    Parameters:
        df: pd.DataFrame to split
        explore: bool. 
            If True -> remove test data and replace numerical values with text.
            If False -> split data into train, validation and test data sets 
        balance: bool. 
            If True -> balance the data set prior to split
            If False -> the data stays unbalanced
        replace: bool. Parameter for balacining data.
            If True -> balance data with upsampling
            If False -> balance data with downsampling
        target: str, target variable.

    '''

    if balance:
        df = balance_data(df, replace=replace)
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    if explore:
        df_explore = replace_values(df_full_train).reset_index(drop=True)
        return df_explore
    else:
        df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

        #df_full_train = df_full_train.reset_index(drop=True)
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        # clean data
        # df_full_train = clean_data(df_full_train)
        df_train = clean_data(df_train)
        df_val = clean_data(df_val)
        df_test = clean_data(df_test)

        y_train = df_train[target].values
        y_val = df_val[target].values
        y_test = df_test[target].values

        del df_train[target]
        del df_val[target]
        del df_test[target]

        return df_train, df_val, df_test, y_train, y_val, y_test

def get_X(train, validate, test):
    ''' 
    Apply One Hot Encoder to ordinal_cat = ['MentHlth', 'PhysHlth'] only. 
    The rest of ordinal categorical columns stay numerical with their range of values from 1 to n
    '''
    # create a one hot encoder that drops the first value, so 1,2,3 encodes as 00, 01, 10
    ohe = OneHotEncoder(handle_unknown='error', drop='first', sparse=False)

    # fit transform train
    X_train = np.concatenate([
        train[binary],
        ohe.fit_transform(train[ordinal_cat]).astype('uint8'),
        train[ordinal_num + numerical].astype('uint8')
    ], axis=1)

    # transform validate and test
    X_validate = np.concatenate([
        validate[binary],
        ohe.transform(validate[ordinal_cat]).astype('uint8'),
        validate[ordinal_num + numerical].astype('uint8')
    ], axis=1)

    X_test = np.concatenate([
        test[binary],
        ohe.transform(test[ordinal_cat]).astype('uint8'),
        test[ordinal_num + numerical].astype('uint8')
    ], axis=1)

    return X_train, X_validate, X_test

def get_X_ohe(train, validate, test, get_features=False):
    ''' 
    Apply Ohe Hot Encoder to all ordinal categorical columns in train, validate and test sets.
    '''
    # create a one hot encoder that drops the first value, so 1,2,3 encodes as 00, 01, 10
    ohe = OneHotEncoder(handle_unknown='error', drop='first', sparse=False)

    # transform BMI to categorical variable
    train = change_bmi(train)
    validate = change_bmi(validate)
    test = change_bmi(test)

    bmi = ["BMI_under", "BMI_over"]

    # fit transform train
    X_train = np.concatenate([
        train[binary],
        ohe.fit_transform(train[ordinal_cat + ordinal_num]).astype('uint8'),
        train[bmi].astype('uint8')
    ], axis=1)

    # transform validate and test
    X_validate = np.concatenate([
        validate[binary],
        ohe.transform(validate[ordinal_cat + ordinal_num]).astype('uint8'),
        validate[bmi].astype('uint8')
    ], axis=1)

    X_test = np.concatenate([
        test[binary],
        ohe.transform(test[ordinal_cat + ordinal_num]).astype('uint8'),
        test[bmi].astype('uint8')
    ], axis=1)
    # pass the list of column names to get names not x0_low, x0_medium, x1_low, x2_medium etc
    ohe_features = ohe.get_feature_names(ordinal_cat + ordinal_num)
    features = binary + list(ohe_features) + bmi
    if get_features:
        return features
    else:
        return X_train, X_validate, X_test