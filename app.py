import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import traceback
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="MediPredict | Diabetes ML",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# UI THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #F3F7FA;
    color: #18324B;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #15354B, #087F8C);
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
}
.hero h1 {
    color: #FFFFFF !important;
    font-size: 36px !important;
}
.hero p {
    color: #E4F5F5 !important;
    font-size: 17px;
}
h1, h2, h3 {
    color: #18324B !important;
}
label, p, span {
    color: #263F56;
}
section[data-testid="stSidebar"] {
    background: #122A40;
}
section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}
div[data-baseweb="input"] input {
    color: #18324B !important;
    background: #FFFFFF !important;
}
div[data-testid="stWidgetLabel"] p {
    color: #18324B !important;
    font-weight: 600 !important;
}
.stButton button {
    background: #087F8C !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    height: 48px;
    font-weight: bold;
}
.stButton button:hover {
    background: #066773 !important;
}
.card {
    background: white;
    border: 1px solid #E1EAF0;
    padding: 22px;
    border-radius: 15px;
    min-height: 115px;
}
.card-label {
    color: #64788C !important;
    font-size: 14px;
}
.card-value {
    color: #18324B !important;
    font-size: 24px;
    font-weight: bold;
    margin-top: 10px;
}
.success-box {
    background: #E6F6EE;
    border: 1px solid #A9DFC3;
    padding: 22px;
    border-radius: 14px;
}
danger-box {
    background: #FFF0EF;
    border: 1px solid #F4C5C1;
    padding: 22px;
    border-radius: 14px;
}
.notice {
    background: #E8F0FC;
    color: #21466C !important;
    padding: 16px;
    border-radius: 12px;
    margin-top: 20px;
    line-height: 1.7;
}
.footer {
    text-align: center;
    color: #718096 !important;
    padding: 25px;
    margin-top: 30px;
    border-top: 1px solid #E1EAF0;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL FILE LOCATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"

# =====================================================
# MODEL LOADER
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find model file.\n"
            f"Expected location: {MODEL_PATH}\n"
            f"Files found: {[f.name for f in BASE_DIR.iterdir()]}"
        )

    # First try pickle
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        return model

    except Exception as pickle_error:

        # Then try joblib
        try:
            model = joblib.load(MODEL_PATH)
            return model

        except Exception as joblib_error:

            raise RuntimeError(
                "Both pickle and joblib loading failed.\n\n"
                f"Pickle error: {pickle_error}\n\n"
                f"Joblib error: {joblib_error}"
            )

# =====================================================
# LOAD MODEL WITH DIAGNOSTICS
# =====================================================

try:

    model = load_model()

except Exception as error:

    st.error("❌ Model Loading Failed")

    st.markdown("### Diagnostic Information")

    st.write("**Expected model path:**")
    st.code(str(MODEL_PATH))

    st.write("**Current application directory:**")
    st.code(str(BASE_DIR))

    st.write("**Files currently available:**")

    try:
        files = [f.name for f in BASE_DIR.iterdir()]
        st.code("\n".join(files))
    except Exception:
        st.write("Could not list files.")

    st.write("**Actual error:**")
    st.code(traceback.format_exc())

    st.info(
        "Check that diabetes_model.pkl is uploaded to the "
        "root of your GitHub repository alongside app.py."
    )

    st.stop()

# =====================================================
# VALIDATE MODEL INPUTS
# =====================================================

if hasattr(model, "n_features_in_"):

    if model.n_features_in_ != 8:

        st.error(
            f"Your model expects {model.n_features_in_} features, "
            "but this app provides 8."
        )

        st.stop()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🩺 MediPredict")

    st.markdown("Machine Learning Health Analytics")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "About Project"]
    )

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.write("**Algorithm**")
    st.caption("Decision Tree Classifier")

    st.write("**Prediction Type**")
    st.caption("Binary Classification")

    st.write("**Input Features**")
    st.caption("8 Parameters")

    st.write("**Model File**")
    st.caption("diabetes_model.pkl")

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🩺 MediPredict</h1>
    <p>Machine Learning Diabetes Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-label">MODEL ALGORITHM</div>
            <div class="card-value">Decision Tree</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-label">INPUT FEATURES</div>
            <div class="card-value">8 Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-label">PREDICTION TYPE</div>
            <div class="card-value">Binary Classification</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Patient Input Parameters")

    st.caption("Enter the parameters below to test your trained model.")

    with st.container(border=True):

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
                value=25
            )

        predict = st.button(
            "✨ Generate Prediction",
            use_container_width=True
        )

    # Input dataframe in training feature order
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

    # =================================================
    # PREDICTION
    # =================================================

    if predict:

        try:

            prediction = model.predict(input_data)[0]

            st.markdown("## Prediction Results")

            if prediction == 1:

                st.markdown("""
                <div style="background:#FFF0EF;
                            border:1px solid #F4C5C1;
                            padding:22px;
                            border-radius:14px;">
                    <h3 style="color:#B42318 !important;">
                        Model Output: Class 1
                    </h3>
                    <p style="color:#7A271A !important;">
                        The model classified the input into Class 1.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div style="background:#E6F6EE;
                            border:1px solid #A9DFC3;
                            padding:22px;
                            border-radius:14px;">
                    <h3 style="color:#067647 !important;">
                        Model Output: Class 0
                    </h3>
                    <p style="color:#05603A !important;">
                        The model classified the input into Class 0.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                class_labels = list(model.classes_)

                prob_map = dict(zip(class_labels, probabilities))

                p0 = float(prob_map.get(0, 0))
                p1 = float(prob_map.get(1, 0))

                a, b = st.columns(2)

                with a:
                    st.metric("Class 0 Probability", f"{p0 * 100:.2f}%")
                    st.progress(p0)

                with b:
                    st.metric("Class 1 Probability", f"{p1 * 100:.2f}%")
                    st.progress(p1)

            with st.expander("View Input Parameters"):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("""
            <div class="notice">
                <strong>Educational Use Only.</strong>
                This machine learning prediction is not a medical diagnosis
                and must not be used for medical decisions.
            </div>
            """, unsafe_allow_html=True)

        except Exception:

            st.error("Prediction failed.")

            st.code(traceback.format_exc())

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About Project":

    st.markdown("## About MediPredict")

    st.write("""
    MediPredict is an educational machine learning application
    demonstrating binary classification using a trained
    Decision Tree model.
    """)

    st.markdown("### Algorithm")

    st.info("""
    Decision Tree Classifier

    A supervised learning algorithm that learns rules from data
    and uses those rules to predict a class for new inputs.
    """)

    st.markdown("### Input Features")

    st.dataframe(
        pd.DataFrame({
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
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="notice">
        Educational demonstration only. Not a medical diagnostic tool.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    MEDIPREDICT · MACHINE LEARNING PROJECT<br>
    Educational Demonstration · Not a Medical Diagnostic Tool
</div>
""", unsafe_allow_html=True)
