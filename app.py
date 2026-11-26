
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# Main title
st.title("🩺 Diabetes Prediction System")

st.write(
    "A Machine Learning application using "
    "Decision Tree Classification."
)

st.info(
    "This application is an educational ML demonstration, "
    "not a medical diagnosis tool."
)

# Sidebar
st.sidebar.title("About Project")

st.sidebar.write("""
### Diabetes Prediction

This application uses a Decision Tree Classifier
to demonstrate how machine learning can classify
data into two categories.

**Algorithm:** Decision Tree

**Dataset:** Diabetes dataset

**Output:**
- Class 0
- Class 1
""")

st.sidebar.markdown("---")
st.sidebar.write("Created using Python, Pandas and Streamlit.")

# Input section
st.subheader("Enter Input Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=300.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=80.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25
        )

    submit = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

# Prediction section
if submit:

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:

        st.warning(
            "The model classified this input as Class 1."
        )

    else:

        st.success(
            "The model classified this input as Class 0."
        )

    st.metric(
        "Model predicted probability for Class 1",
        f"{probabilities[1] * 100:.2f}%"
    )

    st.caption(
        "This percentage is the model's estimated probability, "
        "not a person's actual medical risk."
    )

    st.write("### Input Data")

    st.dataframe(
        input_data,
        use_container_width=True
    )

st.markdown("---")

st.caption(
    "Educational project | Decision Tree Machine Learning | "
    "Prediction is not a medical diagnosis."
)
