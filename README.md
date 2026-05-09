# Heart Disease Risk Predictor

A Streamlit web app that predicts heart disease risk from patient clinical information using a K-Nearest Neighbors machine learning model trained on a heart disease dataset.

> This project is for educational purposes only. It is not a medical diagnosis tool.

## Features

- Interactive Streamlit interface for entering patient data
- KNN classifier trained with scikit-learn
- Preprocessing pipeline for numeric and categorical features
- Grid search for model tuning
- Probability output for heart disease risk
- Saved model loading with `joblib`

## Project Structure

```text
predictingHeartDisease/
+-- app.py                         # Streamlit web interface
+-- train_model.py                 # Trains and saves the model
+-- predict_patient.py             # Simple one-patient prediction test
+-- heart_disease_model.pkl        # Saved trained model
+-- heart_disease_prediction.csv   # Dataset
+-- requirements.txt               # Python dependencies
+-- README.md
```

## How It Works

1. `train_model.py` loads and cleans the dataset.
2. The data is split into training and test sets.
3. Numeric columns are imputed and scaled.
4. Categorical columns are imputed and one-hot encoded.
5. `GridSearchCV` finds the best KNN settings.
6. The best model pipeline is saved to `heart_disease_model.pkl`.
7. `app.py` loads the saved model and predicts risk from user input.

## Model Inputs

The app asks for:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG result
- Maximum heart rate
- Exercise-induced angina
- Oldpeak
- ST slope

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv env
```

On Windows:

```bash
env\Scripts\activate
```

On macOS/Linux:

```bash
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Example Model Performance

The current trained model achieved about 89.67% test accuracy on the local train/test split.

Because this is a small educational dataset, accuracy may vary depending on preprocessing choices, train/test split, and model settings.

## You can try it here
- https://predicting-heart-disease-tnijchjmkk4ow6snek8spa.streamlit.app/


## Disclaimer

This app is for learning and demonstration only. It should not be used to make medical decisions. Always consult a qualified healthcare professional for diagnosis or treatment.
