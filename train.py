import os
import pickle

import pandas as pd 
import numpy as np 
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


seed = 2912
target = "Diabetes_binary"
binary = ['HighBP', 'HighChol', 'CholCheck',  'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'DiffWalk']
binary_and_target = binary + [target]
# binary_and_target = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
# 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 
# 'DiffWalk', 'Diabetes_binary']
ordinal_cat = ['MentHlth', 'PhysHlth']
ordinal_num = ['GenHlth', 'Age', 'Education', 'Income']
#ordinal = ordinal_cat + ordinal_num
bmi = ["BMI_under", "BMI_over"]

# acquire data
dir='data'
filename='diabetis_data.csv'
path_to_file = os.path.join(dir, filename)
if os.path.isfile(path_to_file):
    df = pd.read_csv(path_to_file)
else:
    # fetch dataset 
    from ucimlrepo import fetch_ucirepo 
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

# balance data
x = df[target].value_counts().index[0]
y = df[target].value_counts().index[1]
# Separate majority and minority classes
majority = df[df[target] == x] # majority
minority = df[df[target] == y] # minority

sampled_data = resample(majority, 
                    replace=False,    # sample without replacement
                    n_samples=len(minority), # to match minority class
                    random_state=seed) # reproducible results

# Combine minority class with downsampled majority class
df = pd.concat([sampled_data, minority])

# split data
train, _ = train_test_split(df, test_size=0.2, random_state=seed)
train = train.reset_index(drop=True)

# clean data

#train = train.drop(['Veggies', 'Sex', 'Fruits', 'AnyHealthcare', 'NoDocbcCost'], axis=1)

# reduce number of ordered categories down to three, automatically saves as a category type
train.MentHlth = pd.cut(train.MentHlth, bins=3, labels=['low', 'medium', 'high'])
train.PhysHlth = pd.cut(train.PhysHlth, bins=3, labels=['low', 'medium', 'high'])

# turn ordinar data into categories
ord = ['GenHlth', 'Age', 'Education', 'Income']
for col in ord:
    train[col] = pd.Categorical(train[col]).as_ordered()
# turn binary data and BMI to uint8 (0 to 255)

for col in binary_and_target + ['BMI']:
    train[col] = train[col].astype('uint8')

# change BMI into category
train["BMI_under"] = train.BMI.map(lambda x: 1 if x <= 18 else 0)
train = train.assign(BMI_over = train.BMI.apply(lambda x: 1 if x >= 25 else 0))

# drop the column BMI
# train.drop("BMI", axis=1, inplace=True)

# create y_train
y_train = train[target].values
#del train[target]

# create One Hot Encoder
ohe = OneHotEncoder(handle_unknown='error', drop='first', sparse=False)

# get X_train
# fit transform train
X_train = np.concatenate([
    train[binary],
    ohe.fit_transform(train[ordinal_cat + ordinal_num]).astype('uint8'),
    train[bmi].astype('uint8')
    ], axis=1)

# create and train the model
rf = RandomForestClassifier(
    n_estimators=500, 
    max_depth = 10,
    min_samples_leaf = 10,
    n_jobs=-1, # speed up the process
    random_state=seed
    )
rf.fit(X_train, y_train)

# create deployment directory
dir_ = "deployment"
if not os.path.exists(dir_):
    try:
        os.mkdir(dir_)
    except OSError as e:
        print(e)

# save One Hor Encoder
encoder_file = "encoder.bin"
path_to_encoder = os.path.join(dir_, encoder_file)
with open(path_to_encoder, "wb") as encoder_out:
    pickle.dump(ohe, encoder_out)

# save model
model_file = "model.bin"
path_to_model = os.path.join(dir_, model_file)
with open(path_to_model, "wb") as model_out:
    pickle.dump(rf, model_out)