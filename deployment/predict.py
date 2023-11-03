import requests
import pickle
from flask import Flask, request, jsonify
from pandas import DataFrame
from numpy import select, concatenate

app = Flask("diabetes")

model_file = "model.bin"
encoder_file = "encoder.bin"
transformer_file = "transformer.bin"

binary = ['HighBP', 'HighChol', 'CholCheck',  'Smoker', 'Stroke',
'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'DiffWalk']
ordinal = ['MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']

bmi = ["BMI_under", "BMI_over"]

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

with open(model_file, "rb") as model_in:
    model = pickle.load(model_in)
print("Model loaded")

with open(encoder_file, "rb") as encoder_in:
    ohe = pickle.load(encoder_in)
print("OneHotEncoder loaded")

#@app.route("/predict")
@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json()
    
    p = DataFrame([patient])
    # print("Dataframe created")
    # display(p)
    p.MentHlth = change_values(p, 'MentHlth')
    p.PhysHlth = change_values(p, 'PhysHlth')
    p["BMI_under"] = p.BMI.map(lambda x: 1 if x <= 18 else 0)
    p["BMI_over"] = p.BMI.map(lambda x: 1 if x >=25 else 0)

    X = concatenate([
            p[binary],
            ohe.transform(p[ordinal]).astype('uint8'),
            p[bmi].astype('uint8')
            ], axis=1)
    y = model.predict_proba(X)[:, 1][0].round(2)
    diabetes = y >= 0.4

    diabetes = y >= 0.4  
    result = {
        'diabetes_probability': float(y), 
        'diabetes': bool(diabetes)
    }

    return(jsonify(result))

if __name__=="__main__":
    #app.run(debug=True, host='0.0.0.0', port=2912)
    app.run(debug=True, host='0.0.0.0', port=2912)