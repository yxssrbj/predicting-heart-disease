import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="heart",
    layout="centered",
)


@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")


model = load_model()

st.title("Heart Disease Risk Predictor")
st.markdown(
    "Enter the patient's clinical information below, then click **Predict** "
    "to estimate the model's heart disease risk prediction."
)
st.divider()

with st.form("patient_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    with col2:
        sex = st.selectbox(
            "Sex",
            options=["M", "F"],
            format_func=lambda x: "Male" if x == "M" else "Female",
        )

    st.subheader("Symptoms and History")
    col3, col4 = st.columns(2)
    with col3:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            options=["ASY", "ATA", "NAP", "TA"],
            format_func=lambda x: {
                "ASY": "ASY - Asymptomatic",
                "ATA": "ATA - Atypical Angina",
                "NAP": "NAP - Non-Anginal Pain",
                "TA": "TA - Typical Angina",
            }[x],
        )
    with col4:
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            options=["N", "Y"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
        )

    st.subheader("Clinical Measurements")
    col5, col6, col7 = st.columns(3)
    with col5:
        resting_bp = st.number_input(
            "Resting BP (mmHg)", min_value=1, max_value=300, value=120
        )
    with col6:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", min_value=1, max_value=700, value=200
        )
    with col7:
        max_hr = st.number_input(
            "Max Heart Rate", min_value=50, max_value=250, value=150
        )

    col8, col9 = st.columns(2)
    with col8:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            options=[0, 1],
            format_func=lambda x: "Yes (> 120 mg/dL)" if x == 1 else "No (<= 120 mg/dL)",
        )
    with col9:
        oldpeak = st.number_input(
            "Oldpeak (ST depression)",
            min_value=-5.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            format="%.1f",
        )

    st.subheader("ECG Results")
    col10, col11 = st.columns(2)
    with col10:
        resting_ecg = st.selectbox(
            "Resting ECG",
            options=["Normal", "LVH", "ST"],
            format_func=lambda x: {
                "Normal": "Normal",
                "LVH": "LVH - Left Ventricular Hypertrophy",
                "ST": "ST - ST-T Wave Abnormality",
            }[x],
        )
    with col11:
        st_slope = st.selectbox(
            "ST Slope",
            options=["Up", "Flat", "Down"],
            format_func=lambda x: f"{x} slope",
        )

    st.divider()
    submitted = st.form_submit_button(
        "Predict", use_container_width=True, type="primary"
    )

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "ChestPainType": chest_pain,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "RestingECG": resting_ecg,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    risk_pct = proba[1] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"Heart disease risk detected.\n\n"
            f"The model predicts a {risk_pct:.1f}% probability of heart disease."
        )
    else:
        st.success(
            f"No heart disease risk detected.\n\n"
            f"The model predicts a {risk_pct:.1f}% probability of heart disease."
        )

    st.markdown("**Risk Probability**")
    st.progress(int(risk_pct))
    col_l, col_r = st.columns(2)
    col_l.metric("No Heart Disease", f"{proba[0] * 100:.1f}%")
    col_r.metric("Heart Disease", f"{proba[1] * 100:.1f}%")

    st.caption(
        "This tool is for informational and educational purposes only. "
        "It is not a medical diagnosis. Always consult a qualified healthcare professional."
    )
