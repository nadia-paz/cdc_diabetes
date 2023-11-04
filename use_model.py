import pickle
import os
from src.transform import Transformer

# information about patient
patient = {'HighBP':1,
 'HighChol':1,
 'CholCheck':1,
 'BMI':25,
 'Smoker':0,
 'Stroke':0,
 'HeartDiseaseorAttack':0,
 'PhysActivity':1,
 'Fruits':1,
 'Veggies':1,
 'HvyAlcoholConsump':1,
 'AnyHealthcare':1,
 'NoDocbcCost':0,
 'GenHlth':4,
 'MentHlth':2,
 'PhysHlth':14,
 'DiffWalk':0,
 'Sex':1,
 'Age':6,
 'Education':6,
 'Income':7}

dir_ = "deployment"
model_file = "model.bin"
encoder_file = "encoder.bin"

# load the model
path_to_model = os.path.join(dir_, model_file)
with open(path_to_model, "rb") as model_in:
    model = pickle.load(model_in)

# load One Hot Encoder
path_to_encoder = os.path.join(dir_, encoder_file)
with open(path_to_encoder, "rb") as encoder_in:
    ohe = pickle.load(encoder_in)

t = Transformer(patient, ohe)
X = t.transform_single()

predictions = model.predict_proba(X)[:, 1][0].round(2)

print("Probability of diabetes is", predictions)