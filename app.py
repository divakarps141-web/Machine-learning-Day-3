import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MediPredict | AI Health Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F5F8FC;
}

[data-testid="stSidebar"] {
    background-color: #102A43;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(120deg, #102A43, #0F766E);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
}

.hero p {
    color: #D9F3F0;
    font-size: 15px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #E5ECF3;
    box-shadow: 0 4px 16px rgba(15, 42, 67, 0.04);
}

.metric-card {
    background: white;
    border: 1px solid #E5ECF3;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 10px;
}

.metric-title {
    color: #64748B;
    font-size: 13px;
}

.metric-value {
    color: #102A43;
    font-size: 25px;
    font-weight: 700;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #102A43;
    margin-bottom: 4px;
}

.section-caption {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 20px;
}

div.stButton > button[kind="primary"],
div.stFormSubmitButton > button {
    background: #0F766E;
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #115E59;
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5ECF3;
    padding: 16px;
    border-radius: 14px;
}

input {
    border-radius: 10px !important;
}

.result-card {
    padding: 25px;
    background: white;
    border-radius: 18px;
    border: 1px solid #E5ECF3;
}

.footer {
    text-align: center;
    color: #94A3B8;
    padding-top: 25px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("""
    <div style="padding:10px 0 25px 0;">
        <h1 style="font-size:27px;">🩺 MediPredict</h1>
        <p style="color:#BFD8E8;">AI Health Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "About Project"],
        index=0
    )

    st.markdown("---")

    st.markdown("### Model Information")

    st.markdown("""
    **Algorithm**  
    Decision Tree Classifier

    **Input Features**  
    8 parameters

    **Dataset**  
    Diabetes dataset

    **Framework**  
    Scikit-learn + Streamlit
    """)

    st.markdown("---")

    st.caption("Machine Learning Project · 2026")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Diabetes Prediction Dashboard</h1>
    <p>Machine learning classification demonstration
    using a Decision Tree model.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">MODEL ALGORITHM</div>
            <div class="metric-value">Decision Tree</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">INPUT FEATURES</div>
            <div class="metric-value">08 Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">PREDICTION TYPE</div>
            <div class="metric-value">Binary Classification</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Input Parameters</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Enter values below to explore the model output.'
        '</div>',
        unsafe_allow_html=True
    )

    # ---------------- INPUT FORM ----------------
    with st.form("prediction_form"):

        col1, col2 = st.columns(2, gap="large")

        with col1:

            st.markdown("### 🧬 General Parameters")

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=0
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=25
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

        with col2:

            st.markdown("### 🩺 Additional Parameters")

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.0
            )

            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=1000.0,
                value=80.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=100.0,
                value=20.0
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5
            )

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "✦  Generate Prediction",
            type="primary",
            use_container_width=True
        )

    # ---------------- PREDICTION ----------------
    if submitted:

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

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        res1, res2 = st.columns([1, 1], gap="large")

        with res1:

            if prediction == 1:

                st.warning(
                    "Model Output: Class 1"
                )

            else:

                st.success(
                    "Model Output: Class 0"
                )

            st.metric(
                "Estimated model probability for Class 1",
                f"{probabilities[1] * 100:.2f}%"
            )

            st.progress(float(probabilities[1]))

            st.caption(
                "This probability reflects the model's output "
                "and is not a medical risk estimate."
            )

        with res2:

            st.markdown("""
            <div class="result-card">
                <h3 style="color:#102A43;">
                    📋 Prediction Summary
                </h3>
                <p style="color:#64748B;">
                    The result represents the classification
                    generated by the trained machine learning model.
                </p>
                <p style="color:#64748B;">
                    It is intended for educational demonstration
                    and should not be used to diagnose or make
                    treatment decisions.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("View Submitted Input Data"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

# ---------------- ABOUT PAGE ----------------
elif page == "About Project":

    st.markdown("""
    <div class="card">
        <h2 style="color:#102A43;">About MediPredict</h2>
        <p style="color:#64748B;">
        MediPredict is an educational machine-learning project
        demonstrating binary classification using a Decision
        Tree algorithm and a Streamlit interface.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Workflow")

    st.markdown("""
    1. Dataset loading using Pandas
    2. Data preprocessing
    3. Missing-value handling
    4. Train/test split
    5. Decision Tree model training
    6. Prediction and evaluation
    7. Streamlit deployment
    """)

    st.markdown("### Technology Stack")

    st.write(
        "Python · Pandas · NumPy · Scikit-learn · "
        "Streamlit · Joblib"
    )

    st.info(
        "This application is a student ML demonstration. "
        "It is not a medical diagnostic or decision-support tool."
    )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    MEDIPREDICT · MACHINE LEARNING PROJECT
    <br>
    Educational use only · Not a medical diagnosis
</div>
""", unsafe_allow_html=True)
