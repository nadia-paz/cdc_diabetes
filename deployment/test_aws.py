import requests
import json


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
 
host = "diabetes-env.eba-nddtymey.us-east-1.elasticbeanstalk.com"
url = f'http://{host}/predict'

response = requests.post(url, json=patient) 
result = response.json() 
print(result)