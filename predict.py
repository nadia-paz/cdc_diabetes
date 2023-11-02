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
    t = pickle.load(transformer_in)
print("Transformer loaded")