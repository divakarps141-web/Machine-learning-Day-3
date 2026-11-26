import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import traceback
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediPredict | Diabetes ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM UI THEME
# ============================================================

st.markdown("""
<style>

/* ---------------- GLOBAL ---------------- */

.stApp {
    background-color: #F1F5F9;
    color: #243B53;
    font-family: "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ---------------- TEXT ---------------- */

h1, h2, h3, h4, h5, h6 {
    color: #243B53 !important;
    font-weight: 700 !important;
}

p, li, label {
    color: #42566E !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #42566E;
}

small {
    color: #64748B !important;
}

/* ---------------- HERO HEADER ---------------- */

.hero {
    background: linear-gradient(
        120deg,
        #173B57 0%,
        #176B78 55%,
        #168A8A 100%
    );

    padding: 38px 42px;
    border-radius: 22px;
    margin-bottom: 28px;
    box-shadow: 0 8px 25px rgba(23, 59, 87, 0.10);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -1px;
}

.hero p {
    color: #E1F3F5 !important;
    font-size: 17px !important;
    margin-top: 12px !important;
    margin-bottom: 0 !important;
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background-color: #18334D !important;
}

section[data-testid="stSidebar"] > div {
    background-color: #18334D !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #DCE7F1 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #36536D !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #F1F5F9 !important;
}

/* ---------------- METRIC CARDS ---------------- */

.metric-card {
    background: #FFFFFF;
    border: 1px solid #DCE6EF;
    border-radius: 16px;
    padding: 23px 25px;
    min-height: 145px;
    height: 145px;
    box-shadow: 0 3px 10px rgba(30, 58, 80, 0.035);
}

.metric-label {
    color: #71849A !important;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #243B53 !important;
    font-size: 25px;
    font-weight: 750;
    margin-top: 20px;
    line-height: 1.3;
}

/* ---------------- SECTION HEADINGS ---------------- */

.section-heading {
    color: #243B53 !important;
    font-size: 29px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #71849A !important;
    font-size: 15px;
    margin-bottom: 22px;
}

/* ---------------- INPUT CONTAINER ---------------- */

div[data-testid="stNumberInput"] {
    margin-bottom: 10px;
}

/* Input background */

div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
}

/* Input text */

div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #243B53 !important;
    -webkit-text-fill-color: #243B53 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* Input focus */

div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border: 1px solid #159A9C !important;
    box-shadow: 0 0 0 2px rgba(21, 154, 156, 0.12) !important;
}

/* Input plus/minus buttons */

div[data-testid="stNumberInput"] button {
    background-color: #EAF2F7 !important;
    color: #38556F !important;
    border: none !important;
}

div[data-testid="stNumberInput"] button:hover {
    background-color: #D7E8F1 !important;
    color: #176B78 !important;
}

/* Labels */

div[data-testid="stWidgetLabel"] p {
    color: #344F68 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* ---------------- FORM / INPUT PANEL ---------------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #DCE6EF !important;
    border-radius: 16px !important;
}

/* ---------------- BUTTON ---------------- */

.stButton > button {
    background: linear-gradient(
        90deg,
        #147D86,
        #159A9C
    ) !important;

    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;

    border: none !important;
    border-radius: 11px !important;

    height: 50px !important;

    font-size: 16px !important;
    font-weight: 700 !important;

    transition: 0.2s ease-in-out;
}

.stButton > button:hover {
    background: #116A75 !important;
    border: none !important;
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: scale(0.99);
}

/* ---------------- RESULT CARDS ---------------- */

.result-success {
    background: #E8F5EF;
    border: 1px solid #B8DDCA;
    border-left: 5px solid #26946A;
    border-radius: 14px;
    padding: 22px;
}

.result-success h3 {
    color: #176B4C !important;
    margin-bottom: 8px;
}

.result-success p {
    color: #386B55 !important;
    margin-bottom: 0;
}

.result-warning {
    background: #FFF5E8;
    border: 1px solid #F0D2A9;
    border-left: 5px solid #D99132;
    border-radius: 14px;
    padding: 22px;
}

.result-warning h3 {
    color: #925A16 !important;
    margin-bottom: 8px;
}

.result-warning p {
    color: #805D32 !important;
    margin-bottom: 0;
}

/* ---------------- PROBABILITY ---------------- */

.probability-card {
    background: #FFFFFF;
    border: 1px solid #DCE6EF;
    border-radius: 14px;
    padding: 20px;
}

div[data-testid="stProgress"] > div > div {
    background-color: #159A9C !important;
}

/* ---------------- INFO NOTICE ---------------- */

.notice {
    background: #E8F0F7;
    border: 1px solid #D2E0ED;
    border-radius: 12px;
    padding: 17px 20px;
    margin-top: 20px;
}

.notice p {
    color: #3C5872 !important;
    margin: 0 !important;
    line-height: 1.7;
}

/* ---------------- DATAFRAME ---------------- */

div[data-testid="stDataFrame"] {
    border: 1px solid #DCE6EF;
    border-radius: 12px;
    overflow: hidden;
}

/* ---------------- EXPANDER ---------------- */

div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE6EF !important;
    border-radius: 12px !important;
}

div[data-testid="stExpander"] summary {
    color: #344F68 !important;
}

/* ---------------- ALERTS ---------------- */

div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------------- FOOTER ---------------- */

.footer {
    text-align: center;
    color: #8496A8 !important;
    font-size: 13px;
    padding-top: 30px;
    margin-top: 35px;
    border-top: 1px solid #DCE6EF;
    line-height: 1.8;
}

/* ---------------- RESPONSIVE ---------------- */

@media (max-width: 768px) {

    .hero {
        padding: 25px;
    }

    .hero h1 {
        font-size: 30px !important;
    }

    .metric-card {
        height: auto;
        min-height: 115px;
    }

    .metric-value {
        font-size: 21px;
    }

    .section-heading {
        font-size: 24px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL FILE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    try:

        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)

    except Exception as pickle_error:

        try:
            return joblib.load(MODEL_PATH)

        except Exception as joblib_error:

            raise RuntimeError(
                f"Pickle error: {pickle_error}\n"
                f"Joblib error: {joblib_error}"
            )


try:

    model = load_model()

except Exception as error:

    st.error("Model could not be loaded.")

    st.markdown("### Diagnostic Information")

    st.code(str(MODEL_PATH))

    st.markdown("### Files Available")

    try:
        st.code("\n".join([f.name for f in BASE_DIR.iterdir()]))
    except Exception:
        st.write("Unable to list files.")

    st.markdown("### Actual Error")

    st.code(traceback.format_exc())

    st.stop()


# ============================================================
# MODEL VALIDATION
# ============================================================

if hasattr(model, "n_features_in_"):

    if model.n_features_in_ != 8:

        st.error(
            f"Model expects {model.n_features_in_} input features. "
            "This application supplies 8 features."
        )

        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding-top:20px;padding-bottom:10px;">
            <h2 style="color:#F8FAFC !important;">
                🩺 MediPredict
            </h2>
            <p style="color:#B8CADB !important;">
                Machine Learning Health Analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "About Project"],
        label_visibility="visible"
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


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🩺 MediPredict</h1>
    <p>Machine Learning Diabetes Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MODEL ALGORITHM</div>
            <div class="metric-value">Decision Tree</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">INPUT FEATURES</div>
            <div class="metric-value">8 Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">PREDICTION TYPE</div>
            <div class="metric-value">Binary Classification</div>
        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Patient Input Parameters</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter the input values to explore your trained ML model.'
        '</div>',
        unsafe_allow_html=True
    )


    with st.container(border=True):

        col1, col2 = st.columns(2, gap="large")

        # LEFT COLUMN

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

            glucose = st.number_input(
                "Glucose",
                min_value=0.0,
                max_value=300.0,
                value=120.0,
                step=1.0
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


        # RIGHT COLUMN

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

            pedigree = st.number_input(
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


        st.markdown("<br>", unsafe_allow_html=True)

        predict_button = st.button(
            "✨ Generate Prediction",
            use_container_width=True
        )


    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

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

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age
        ]],
        columns=feature_names
    )


    # --------------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------------

    if predict_button:

        try:

            prediction = model.predict(input_data)[0]

            st.markdown(
                '<div class="section-heading">Prediction Results</div>',
                unsafe_allow_html=True
            )

            # CLASS 0

            if prediction == 0:

                st.markdown("""
                <div class="result-success">
                    <h3>Model Output: Class 0</h3>
                    <p>
                        The model classified the supplied input
                        into Class 0.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # CLASS 1

            else:

                st.markdown("""
                <div class="result-warning">
                    <h3>Model Output: Class 1</h3>
                    <p>
                        The model classified the supplied input
                        into Class 1.
                    </p>
                </div>
                """, unsafe_allow_html=True)


            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                class_labels = list(model.classes_)

                probability_map = dict(
                    zip(class_labels, probabilities)
                )

                class0_probability = float(
                    probability_map.get(0, 0)
                )

                class1_probability = float(
                    probability_map.get(1, 0)
                )


                st.markdown("<br>", unsafe_allow_html=True)

                prob_col1, prob_col2 = st.columns(2, gap="medium")


                with prob_col1:

                    st.markdown("""
                    <div class="probability-card">
                        <p style="color:#71849A !important;">
                            Class 0 Probability
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric(
                        label="",
                        value=f"{class0_probability * 100:.2f}%"
                    )

                    st.progress(class0_probability)


                with prob_col2:

                    st.markdown("""
                    <div class="probability-card">
                        <p style="color:#71849A !important;">
                            Class 1 Probability
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric(
                        label="",
                        value=f"{class1_probability * 100:.2f}%"
                    )

                    st.progress(class1_probability)


            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            with st.expander("View Input Parameters"):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )


            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            st.markdown("""
            <div class="notice">
                <p>
                    <strong>Educational Demonstration:</strong>
                    This project demonstrates machine learning
                    classification. It is not a medical diagnostic
                    tool. Model predictions should not be used
                    to make medical decisions.
                </p>
            </div>
            """, unsafe_allow_html=True)


        except Exception:

            st.error("Prediction failed.")

            st.code(traceback.format_exc())


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-heading">About MediPredict</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    MediPredict is an educational machine learning project
    designed to demonstrate diabetes dataset classification
    using a trained Decision Tree model.
    """)

    st.markdown("### Project Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MODEL</div>
            <div class="metric-value">Decision Tree Classifier</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">TASK</div>
            <div class="metric-value">Binary Classification</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("### Input Features")

    feature_table = pd.DataFrame({
        "Feature": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("### How It Works")

    st.markdown("""
    1. Enter the eight input parameters.
    2. The application sends the values to the trained model.
    3. The model predicts a classification.
    4. The application displays the predicted class and
       probabilities when available.
    """)


    st.markdown("""
    <div class="notice">
        <p>
            Educational use only. This application is not
            intended for diagnosis, treatment, or medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <strong>MEDIPREDICT</strong> · MACHINE LEARNING PROJECT<br>
    Educational Demonstration · Not a Medical Diagnostic Tool
</div>
""", unsafe_allow_html=True)
