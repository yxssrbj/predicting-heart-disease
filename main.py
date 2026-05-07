import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("heart_disease_prediction.csv")
df.head()

print(df.dtypes)
df.dtypes.value_counts()
df.describe()
df.isna().sum()
df.describe(include=["object"])
df["FastingBS"].unique()
df["HeartDisease"].unique()
categorical_cols = [
    "Sex",
    "ChestPainType",
    "FastingBS",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
    "HeartDisease",
]


fig = plt.figure(figsize=(16, 15))

for idx, col in enumerate(categorical_cols):
    ax = plt.subplot(4, 2, idx + 1)

    sns.countplot(data=df, x=col, ax=ax)

    ax.set_title(col)

    for container in ax.containers:
        ax.bar_label(container, label_type="center")


fig = plt.figure(figsize=(16, 15))
for idx, col in enumerate(categorical_cols):
    ax = plt.subplot(4, 2, idx + 1)
    sns.countplot(x=df[col], hue=df["HeartDisease"], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, label_type="center")


print("Zero values in RestingBP:", (df["RestingBP"] == 0).sum())
print("Zero values in Cholesterol:", (df["Cholesterol"] == 0).sum())

df_clean = df.copy()
df_clean = df_clean[df_clean["RestingBP"] != 0]

no_heart_disease_mask = df_clean["HeartDisease"] == 0

## replacing zero values with the median of non zero values of people how were diagnosed with heart disease
#
cholesterol_with_heart_disease = df_clean.loc[~no_heart_disease_mask, "Cholesterol"]
cholesterol_without_heart_disease = df_clean.loc[no_heart_disease_mask, "Cholesterol"]

df_clean.loc[no_heart_disease_mask, "Cholesterol"] = (
    cholesterol_without_heart_disease.replace(
        to_replace=0, value=cholesterol_without_heart_disease.median()
    )
)
df_clean.loc[~no_heart_disease_mask, "Cholesterol"] = (
    cholesterol_with_heart_disease.replace(
        to_replace=0, value=cholesterol_with_heart_disease.median()
    )
)
print("Values after replacement -----")
print(df_clean[["Cholesterol", "RestingBP"]].describe())

df_clean = pd.get_dummies(df_clean, drop_first=True)
print(df_clean.head())

plt.tight_layout()
plt.show()
