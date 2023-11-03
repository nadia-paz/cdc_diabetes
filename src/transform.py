from pandas import DataFrame
from numpy import select, concatenate
import pickle

from sklearn.preprocessing import OneHotEncoder
#from sklearn.base import BaseEstimator, TransformerMixin



class Transformer():

    def __init__(self, patient:dict, ohe: OneHotEncoder):
        self.patient = patient
        self.ohe = ohe

    def get_encoder(self):
        print("inside get_encoder")
        return self.ohe

    def get_dict(self):
        print("inside get_dict")
        return self.patient

    # create a function that applies custom transormation
    def transform_single(self):
        '''
        Apply custom transformation.
        Parameters:
            patient: dictionary with respondent's data
            ohe: OneHotEncoder
        '''
        binary = ['HighBP', 'HighChol', 'CholCheck',  'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'DiffWalk']
        ordinal = ['MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']
        
        bmi = ["BMI_under", "BMI_over"]

        ohe = self.ohe
        # print("Encoder saved")
        patient = self.patient
        # print("Dictionary saved")
    
        def change_values(x, target):
            ''' 
            Function to replace values in MentHlth and PhysHlth columns
            below 10 days -> low, above 20 days -> highbb.;v , everything else "medium"
            '''
            # print("inside change values func")
            # print(f"x: -> {x}, target: -> {target}")
            conds = [x[target] <= 10, x[target] > 20]
            choices = ['low', 'high']
            return select(conds, choices, default='medium').astype("object")
        
        p = DataFrame([patient])
        # print("Dataframe created")
        # display(p)
        p.MentHlth = change_values(p, 'MentHlth')
        p.PhysHlth = change_values(p, 'PhysHlth')
        p["BMI_under"] = p.BMI.map(lambda x: 1 if x <= 18 else 0)
        p["BMI_over"] = p.BMI.map(lambda x: 1 if x >=25 else 0)
        #p.drop("BMI", axis=1, inplace=True)
        
        return concatenate([
            p[binary],
            ohe.transform(p[ordinal]).astype('uint8'),
            p[bmi].astype('uint8')
            ], axis=1)
