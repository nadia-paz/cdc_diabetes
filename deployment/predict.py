import requests
import pickle
from flask import Flask, request, jsonify

app = Flask("diabetes")

model_file = "model.bin"
encoder_file = "encoder.bin"
transformer_file = "transformer.bin"

with open(model_file, "rb") as model_in:
    model = pickle.load(model_in)
print("Model loaded")

with open(encoder_file, "rb") as encoder_in:
    ohe = pickle.load(encoder_in)
print("OneHotEncoder loaded")

with open(transformer_file, "rb") as transformer_in:
    Transformer = pickle.load(transformer_in)
print("Transformer loaded")

#@app.route("/predict")
@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json()
    
    t = Transformer(patient, ohe)

    X = t.transform_single()
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
    app.run(debug=True)