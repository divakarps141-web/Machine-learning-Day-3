import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MediPredict | Diabetes ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ADVANCED UI DESIGN
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4F8FB;
    color: #172B4D;
    font-family: 'Inter', sans-serif;
}

/* Main content width */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #10243A;
}

section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}

/* Main text */
h1, h2, h3, h4 {
    color: #17324D !important;
}

p, label, span {
    color: inherit;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #12324A, #087F8C);
    padding: 35px;
    border-radius: 22px;
    margin-bottom: 25px;
    color: white !important;
    box-shadow: 0 8px 25px rgba(16, 36, 58, 0.10);
}

.hero h1 {
    color: white !important;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #E0F2F1 !important;
    font-size: 17px;
    margin-bottom: 0;
}

/* Cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E1EAF0;
    border-radius: 16px;
    padding: 22px;
    height: 145px;
    box-shadow: 0 4px 15px rgba(16, 36, 58, 0.04);
}

.card-title {
    color: #62758A !important;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

.card-value {
    color: #17324D !important;
    font-size: 27px;
    font-weight: 750;
}

/* Section card */
.section-card {
    background: white;
    border: 1px solid #E1EAF0;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
    margin-bottom: 20px;
}

/* Input labels */
div[data-testid="stWidgetLabel"] p {
    color: #243B53 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Input boxes */
div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] input {
    color: #172B4D !important;
    background-color: #FFFFFF !important;
}

/* Select boxes */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #087F8C, #0A9D9A);
    color: #FFFFFF !important;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #066B75;
    color: white !important;
    border: none;
    transform: translateY(-1px);
}

/* Result boxes */
.result-positive {
    background: #FFF1F0;
    border: 1px solid #FFD6D2;
    border-radius: 16px;
    padding: 24px;
}

.result-negative {
    background: #EAF8F1;
    border: 1px solid #B8E7CF;
    border-radius: 16px;
    padding: 24px;
}

.result-heading {
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-description {
    font-size: 15px;
    line-height: 1.7;
}

/* Notice */
.notice {
    background: #EAF2FF;
    border: 1px solid #C8DBFA;
    border-radius: 12px;
    padding: 16px 20px;
    color: #23456B !important;
    font-size: 14px;
    line-height: 1.7;
    margin-top: 20px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 25px 10px;
    color: #718096 !important;
    font-size: 13px;
    border-top: 1px solid #E1EAF0;
    margin-top: 35px;
}

/* Progress */
div[data-testid="stProgressBar"] > div > div {
    background-color: #0A9D9A;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border: 1px solid #E1EAF0;
    border-radius: 16px;
}

/* Hide default menu */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Responsive */
@media screen and (max-width: 768px) {
    .hero {
        padding: 25px;
    }

    .hero h1 {
        font-size: 28px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL LOADING
# =========================================================

MODEL_PATH = Path(__file__).parent / "diabetes_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as error:
    model_loaded = False
    model = None

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:20px 5px 15px 5px;">
        <div style="font-size:42px;">🩺</div>
        <h2 style="color:white !important;margin-top:10px;">
            MediPredict
        </h2>
        <p style="color:#B8C7D9 !important;">
            Machine Learning Health Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "About Project"],
        index=0
    )

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.markdown("**Algorithm**")
    st.caption("Decision Tree Classifier")

    st.markdown("**Prediction Type**")
    st.caption("Binary Classification")

    st.markdown("**Input Features**")
    st.caption("8 Parameters")

    st.markdown("**Model File**")
    st.caption("diabetes_model.pkl")

    st.divider()

    st.markdown("""
    <div style="font-size:12px;color:#AFC1D5;">
        Educational ML Project<br>
        Version 1.0
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🩺 MediPredict</h1>
    <p>Machine Learning Diabetes Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MODEL CHECK
# =========================================================

if not model_loaded:

    st.error(
        "Model file could not be loaded. "
        "Make sure diabetes_model.pkl is present in the same folder as app.py."
    )

    st.stop()

# Check number of model features
if hasattr(model, "n_features_in_"):

    if model.n_features_in_ != 8:

        st.error(
            f"Your saved model expects {model.n_features_in_} input features, "
            "but this application provides 8 features. "
            "Please check how the model was trained."
        )

        st.stop()

# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    # Top information cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">MODEL ALGORITHM</div>
            <div class="card-value">Decision Tree</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">INPUT FEATURES</div>
            <div class="card-value">08 Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">PREDICTION TYPE</div>
            <div class="card-value">Binary Classification</div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # INPUT SECTION
    # =====================================================

    st.markdown("""
    <div style="margin-top:35px;">
        <h2>Patient Input Parameters</h2>
        <p style="color:#62758A;">
            Enter the input values to generate a model prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):

        st.markdown("### 🧬 General Parameters")

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="Number of pregnancies"
            )

            glucose = st.number_input(
                "Glucose",
                min_value=0.0,
                max_value=300.0,
                value=120.0,
                step=1.0,
                help="Glucose measurement"
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=0.0,
                max_value=200.0,
                value=80.0,
                step=1.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0
            )

        with col2:

            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=1000.0,
                value=80.0,
                step=1.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=0.1
            )

            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.50,
                step=0.01,
                format="%.2f"
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=25,
                step=1
            )

        st.markdown("---")

        predict_button = st.button(
            "✨ Generate Prediction",
            use_container_width=True
        )

    # =====================================================
    # FEATURE ORDER
    # =====================================================

    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    input_values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]

    input_data = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        try:

            prediction = model.predict(input_data)[0]

            st.markdown("""
            <div style="margin-top:30px;">
                <h2>Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)

            # Probability calculation
            probabilities = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                classes = list(model.classes_)

                probability_map = {
                    classes[i]: float(probabilities[i])
                    for i in range(len(classes))
                }

                class_0_probability = probability_map.get(0, 0.0)
                class_1_probability = probability_map.get(1, 0.0)

            else:

                class_0_probability = 0.0
                class_1_probability = 0.0

            # Result display
            if prediction == 1:

                st.markdown("""
                <div class="result-positive">
                    <div class="result-heading" style="color:#B42318;">
                        Model Output: Class 1
                    </div>
                    <div class="result-description" style="color:#7A271A;">
                        The model classified the supplied input values
                        into Class 1.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-negative">
                    <div class="result-heading" style="color:#067647;">
                        Model Output: Class 0
                    </div>
                    <div class="result-description" style="color:#05603A;">
                        The model classified the supplied input values
                        into Class 0.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability cards
            if probabilities is not None:

                st.markdown("<br>", unsafe_allow_html=True)

                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:

                    st.metric(
                        label="Class 0 Probability",
                        value=f"{class_0_probability * 100:.2f}%"
                    )

                    st.progress(class_0_probability)

                with prob_col2:

                    st.metric(
                        label="Class 1 Probability",
                        value=f"{class_1_probability * 100:.2f}%"
                    )

                    st.progress(class_1_probability)

            # Input summary
            with st.expander("View Input Parameters"):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

            # Educational disclaimer
            st.markdown("""
            <div class="notice">
                <strong>Educational Demonstration:</strong>
                This application demonstrates a machine learning
                classification model. It is not a medical diagnostic tool.
                The prediction is not a diagnosis and must not be used
                to make medical decisions.
            </div>
            """, unsafe_allow_html=True)

        except Exception as error:

            st.error("Prediction failed.")

            st.code(str(error))

# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.markdown("## About MediPredict")

    st.markdown("""
    MediPredict is a machine learning project designed to demonstrate
    binary classification using a Decision Tree Classifier.

    The application uses a previously trained machine learning model
    stored as `diabetes_model.pkl`. The user enters eight input
    parameters, and the model generates a prediction.
    """)

    st.markdown("### Machine Learning Algorithm")

    st.info("""
    **Decision Tree Classifier**

    A Decision Tree is a supervised machine learning algorithm
    that learns decision rules from training data and uses those
    rules to classify new input data.
    """)

    st.markdown("### Input Features")

    features = pd.DataFrame({
        "Feature": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ],
        "Description": [
            "Pregnancy count",
            "Glucose measurement",
            "Blood pressure measurement",
            "Skin thickness measurement",
            "Insulin measurement",
            "Body mass index",
            "Diabetes pedigree function",
            "Age in years"
        ]
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Model Output")

    st.markdown("""
    - **Class 0:** Model prediction for class zero.
    - **Class 1:** Model prediction for class one.
    - **Probability:** Model-estimated probability for each class,
      when supported by the saved model.
    """)

    st.markdown("""
    <div class="notice">
        This project is developed for learning, experimentation,
        and academic demonstration of machine learning.
        It must not be used as a substitute for professional healthcare.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    MEDIPREDICT · MACHINE LEARNING PROJECT<br>
    Educational Demonstration · Not a Medical Diagnostic Tool
</div>
""", unsafe_allow_html=True)
