import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# 1. Load the dataset
# Each row is one patient. HeartDisease is the answer we want the model to learn.
df = pd.read_csv("heart_disease_prediction.csv")


# 2. Basic cleaning
# RestingBP = 0 is not realistic, so we remove those rows.
df = df[df["RestingBP"] != 0].copy()

# Cholesterol = 0 is also not realistic. We replace it with the median non-zero cholesterol.
cholesterol_median = df.loc[df["Cholesterol"] != 0, "Cholesterol"].median()
df.loc[df["Cholesterol"] == 0, "Cholesterol"] = cholesterol_median


# 3. Split the dataset into features X and target y
# X = the patient information we use to predict
# y = the answer: 0 means no heart disease, 1 means heart disease
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]


# 4. Split into training and test data
# The model learns from the training data.
# The test data checks how well it works on patients it has not seen before.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=417,
    stratify=y,
)


# 5. Tell scikit-learn which columns are numeric and which are categorical
numeric_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


# 6. Build preprocessing steps
# Numeric columns: fill missing values with the median, then scale between 0 and 1.
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

# Categorical columns: fill missing values, then turn text categories into numbers.
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# ColumnTransformer applies the right preprocessing to the right columns.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# 7. Build the full model pipeline
# This means: preprocess the data first, then train KNN.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier()),
    ]
)


# 8. Use grid search to find good KNN settings
# The parameter names use classifier__ because KNN is inside the pipeline under that name.
params = {
    "classifier__n_neighbors": range(1, 21),
    "classifier__metric": ["manhattan", "minkowski"],
    "classifier__weights": ["uniform", "distance"],
}

grid = GridSearchCV(
    model,
    params,
    scoring="accuracy",
    cv=5,
)

grid.fit(X_train, y_train)


# 9. Evaluate the best model on the test set
best_model = grid.best_estimator_
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Best settings:")
print(grid.best_params_)

print()
print("Test accuracy:")
print(f"{accuracy * 100:.2f}%")

print()
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print()
print("Classification report:")
print(classification_report(y_test, predictions))


# 10. Save the trained model
# This file can be loaded later by an app without retraining.
joblib.dump(best_model, "heart_disease_model.pkl")
print()
print("Saved model to heart_disease_model.pkl")
