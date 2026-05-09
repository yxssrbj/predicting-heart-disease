import joblib
import pandas as pd


# Load the model that was created by train_model.py.
# This model already includes preprocessing plus the trained KNN classifier.
model = joblib.load("heart_disease_model.pkl")


# This is one example patient.
# The column names must match the original training dataset, except HeartDisease is not included
# because that is what we want the model to predict.
patient = pd.DataFrame([
    {
        "Age": 54,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 130,
        "Cholesterol": 250,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 1.2,
        "ST_Slope": "Flat",
    }
])


# predict() returns the class label: 0 or 1.
prediction = model.predict(patient)[0]

# predict_proba() returns probabilities for both classes.
# Index 0 is probability of no heart disease.
# Index 1 is probability of heart disease.
probabilities = model.predict_proba(patient)[0]
heart_disease_probability = probabilities[1]


if prediction == 1:
    print("Prediction: Heart disease likely")
else:
    print("Prediction: Heart disease not likely")

print(f"Probability of heart disease: {heart_disease_probability * 100:.2f}%")
