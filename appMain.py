from flask import Flask, render_template, request, jsonify
import pyttsx3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import csv
import re
import warnings

# Initialize Flask app
app = Flask(__name__)

# Load models and data
training = pd.read_csv("Data/Training.csv")
cols = training.columns[:-1]
x = training[cols]
y = training["prognosis"]

# Train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(x, y)

# Load additional data
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def getDescription():
    global description_list
    with open("MasterData/symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getSeverityDict():
    global severityDictionary
    with open("MasterData/Symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            severityDictionary[row[0]] = int(row[1])

def getprecautionDict():
    global precautionDictionary
    with open("MasterData/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# Example function to process symptoms and predict
def process_symptoms(symptoms_exp):
    input_vector = np.zeros(len(cols))
    for symptom in symptoms_exp:
        if symptom in cols:
            input_vector[cols.get_loc(symptom)] = 1
    return clf.predict([input_vector])

# Web UI for input and output
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    symptoms_exp = request.form.getlist('symptoms')  # Get symptoms from form input
    prediction = process_symptoms(symptoms_exp)
    
    # You can add more logic here to return additional data like description, precautions, etc.
    result_message = f"Predicted disease: {prediction[0]}"
    description = description_list.get(prediction[0], "No description available.")
    
    return jsonify({"result_message": result_message, "description": description})

# Start the app
if __name__ == '__main__':
    app.run(debug=True)
    getDescription()
    getSeverityDict()
    getprecautionDict()
