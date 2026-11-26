
import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# 1. PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="MediPredict | Diabetes ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. CUSTOM CSS - CLEAN & CONSISTENT UI
# ==========================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hide Streamlit top decoration */
[data-testid="stHeader"] {
    background: transparent;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background: #14263D;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

[data-testid="stSidebar"] hr {
    border-color: #344A62;
}

.sidebar-brand {
    padding: 10px 5px 20px 5px;
}

.brand-icon {
    font-size: 34px;
    margin-bottom: 10px;
}

.brand-name {
    color: #FFFFFF !important;
    font-size: 27px;
    font-weight: 800;
    margin: 0;
}

.brand-subtitle {
    color: #B7C9DC !important;
    font-size: 13px;
    margin-top: 5px;
}

.sidebar-heading {
    color: #9FB4CA !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Sidebar radio navigation */

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #E8F0F8 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] p {
    color: #E8F0F8 !important;
}

/* ---------- HERO ---------- */

.hero {
    background: linear-gradient(120deg, #14263D 0%, #15536A 55%, #0D9488 100%);
    padding: 36px 38px;
    border-radius: 22px;
    margin-bottom: 24px;
}

.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 30px;
    background: rgba(255,255,255,0.12);
    color: #D8FFFA !important;
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.hero-title {
    color: #FFFFFF !important;
    font-size: 34px;
    font-weight: 800;
    margin-top: 17px;
    margin-bottom: 10px;
    line-height: 1.3;
}

.hero-description {
    color: #E0F2FE !important;
    font-size: 14px;
    line-height: 1.7;
    margin: 0;
}

/* ---------- SECTION TITLES ---------- */

.section-title {
    color: #14263D !important;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-description {
    color: #64748B !important;
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 18px;
}

/* ---------- SUMMARY CARDS ---------- */

.summary-card {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.025);
}

.summary-label {
    color: #64748B !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.7px;
    text-transform: uppercase;
}

.summary-value {
    color: #14263D !important;
    font-size: 21px;
    font-weight: 750;
    margin-top: 12px;
}

/* ---------- FORM CONTAINER ---------- */

.form-box {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 18px;
    padding: 22px 24px 10px 24px;
    box-shadow: 0 3px 14px rgba(15,23,42,0.025);
}

/* ---------- INPUT LABELS ---------- */

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stNumberInput label {
    color: #334155 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* ---------- INPUT BOXES ---------- */

[data-testid="stNumberInput"] input {
    color: #14263D !important;
    -webkit-text-fill-color: #14263D !important;
    background-color: #FFFFFF !important;
    font-weight: 500 !important;
}

[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D7E0EA !important;
    border-radius: 10px !important;
}

[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border-color: #0D9488 !important;
    box-shadow: 0 0 0 2px rgba(13,148,136,0.10) !important;
}

[data-testid="stNumberInput"] button {
    background: #F8FAFC !important;
}

[data-testid="stNumberInput"] button svg {
    fill: #475569 !important;
}

/* ---------- SUBMIT BUTTON ---------- */

[data-testid="stFormSubmitButton"] button {
    width: 100%;
    height: 48px;
    background: #0D9488 !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
    transition: 0.2s ease;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: #0F766E !important;
    border: 0 !important;
    box-shadow: 0 5px 15px rgba(13,148,136,0.15);
}

[data-testid="stFormSubmitButton"] button p {
    color: #FFFFFF !important;
}

/* ---------- RESULT CONTAINERS ---------- */

.result-box {
    border-radius: 17px;
    padding: 23px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
    min-height: 180px;
}

.result-class-zero {
    background: #F0FDFA;
    border: 1px solid #99F6E4;
}

.result-class-one {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
}

.result-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.7px;
    color: #64748B !important;
}

.result-heading {
    font-size: 23px;
    font-weight: 800;
    margin-top: 12px;
    margin-bottom: 10px;
}

.result-text {
    font-size: 13px;
    color: #475569 !important;
    line-height: 1.7;
}

/* ---------- METRICS ---------- */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 14px;
    padding: 17px 18px;
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricLabel"] p {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #14263D !important;
}

[data-testid="stMetricValue"] div {
    color: #14263D !important;
}

/* ---------- EXPANDER ---------- */

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    padding: 14px 18px !important;
    background: #FFFFFF !important;
    color: #14263D !important;
    min-height: 52px !important;
}

[data-testid="stExpander"] summary p {
    color: #14263D !important;
    font-weight: 650 !important;
    font-size: 14px !important;
    margin: 0 !important;
}

[data-testid="stExpander"] summary svg {
    color: #0D9488 !important;
    flex-shrink: 0 !important;
}

[data-testid="stExpander"] details {
    background: #FFFFFF !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
    border-top: 1px solid #E5EAF0 !important;
    padding: 15px !important;
}

/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- ALERTS ---------- */

[data-testid="stAlert"] {
    border-radius: 12px !important;
}

[data-testid="stAlert"] p {
    line-height: 1.65 !important;
}

/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #94A3B8 !important;
    font-size: 11px;
    line-height: 1.8;
    padding-top: 25px;
}

/* ---------- RESPONSIVE ---------- */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .hero {
        padding: 24px;
        border-radius: 17px;
    }

    .hero-title {
        font-size: 26px;
    }

    .summary-value {
        font-size: 18px;
    }

    .form-box {
        padding: 15px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load("diabetes_model.pkl")

    return model


try:
    model = load_model()
    model_loaded = True

except Exception as error:
    model = None
    model_loaded = False


# ==========================================================
# 4. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🩺</div>
        <p class="brand-name">MediPredict</p>
        <p class="brand-subtitle">Machine Learning Health Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<p class="sidebar-heading">NAVIGATION</p>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        ["Dashboard", "About Project"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<p class="sidebar-heading">MODEL INFORMATION</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    **Algorithm**

    Decision Tree Classifier

    **Prediction Type**

    Binary Classification

    **Input Parameters**

    8 Features
    """)

    st.divider()

    if model_loaded:
        st.success("Model loaded successfully")
    else:
        st.error("Model file not found")

    st.caption("Educational ML Project")


# ==========================================================
# 5. HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">

    <span class="hero-badge">
        MACHINE LEARNING PROJECT
    </span>

    <h1 class="hero-title">
        Diabetes Prediction Dashboard
    </h1>

    <p class="hero-description">
        A clean interactive interface for exploring a Decision Tree
        classification model using eight input parameters.
    </p>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# 6. DASHBOARD PAGE
# ==========================================================

if page == "Dashboard":

    # ---------- SUMMARY CARDS ----------

    card1, card2, card3 = st.columns(3, gap="medium")

    with card1:
        st.markdown("""
        <div class="summary-card">
            <div class="summary-label">MODEL ALGORITHM</div>
            <div class="summary-value">Decision Tree</div>
        </div>
        """, unsafe_allow_html=True)

    with card2:
        st.markdown("""
        <div class="summary-card">
            <div class="summary-label">INPUT FEATURES</div>
            <div class="summary-value">8 Parameters</div>
        </div>
        """, unsafe_allow_html=True)

    with card3:
        st.markdown("""
        <div class="summary-card">
            <div class="summary-label">OUTPUT TYPE</div>
            <div class="summary-value">Binary Classification</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ---------- INPUT SECTION ----------

    st.markdown(
        '<div class="section-title">Input Parameters</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="section-description">
        Enter the values below to explore the trained model's output.
    </div>
    """, unsafe_allow_html=True)

    if model_loaded:

        with st.form("diabetes_prediction_form"):

            # Two equal columns
            left_column, right_column = st.columns(
                2,
                gap="large"
            )

            # ---------- LEFT COLUMN ----------

            with left_column:

                st.markdown("#### General Parameters")

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

            # ---------- RIGHT COLUMN ----------

            with right_column:

                st.markdown("#### Additional Parameters")

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

            st.write("")

            submit = st.form_submit_button(
                "Generate Prediction",
                use_container_width=True
            )

        # ==================================================
        # 7. PREDICTION RESULT
        # ==================================================

        if submit:

            # Maintain the exact feature order used during training
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
                    diabetes_pedigree,
                    age
                ]],
                columns=feature_names
            )

            try:

                prediction = model.predict(input_data)[0]

                # Safely handle model probability support
                probability_available = hasattr(model, "predict_proba")

                if probability_available:

                    probability_values = model.predict_proba(input_data)[0]

                    model_classes = list(model.classes_)

                    probability_map = {
                        model_classes[i]: float(probability_values[i])
                        for i in range(len(model_classes))
                    }

                    probability_0 = probability_map.get(0, 0.0)
                    probability_1 = probability_map.get(1, 0.0)

                else:

                    probability_0 = None
                    probability_1 = None

                st.write("")

                st.markdown(
                    '<div class="section-title">Prediction Results</div>',
                    unsafe_allow_html=True
                )

                st.markdown("""
                <div class="section-description">
                    Results generated by the trained classification model.
                </div>
                """, unsafe_allow_html=True)

                # ---------- PREDICTION CARD ----------

                result_col, probability_col = st.columns(
                    [1.15, 1],
                    gap="large"
                )

                with result_col:

                    if prediction == 0:

                        st.markdown("""
                        <div class="result-box result-class-zero">

                            <div class="result-label">
                                MODEL CLASSIFICATION
                            </div>

                            <div class="result-heading"
                            style="color:#0F766E !important;">
                                Class 0
                            </div>

                            <div class="result-text">
                                The model predicted Class 0 for the
                                supplied input parameters.
                            </div>

                        </div>
                        """, unsafe_allow_html=True)

                    else:

                        st.markdown("""
                        <div class="result-box result-class-one">

                            <div class="result-label">
                                MODEL CLASSIFICATION
                            </div>

                            <div class="result-heading"
                            style="color:#C2410C !important;">
                                Class 1
                            </div>

                            <div class="result-text">
                                The model predicted Class 1 for the
                                supplied input parameters.
                            </div>

                        </div>
                        """, unsafe_allow_html=True)

                # ---------- PROBABILITY CARD ----------

                with probability_col:

                    if probability_available:

                        st.metric(
                            "Class 1 Probability",
                            f"{probability_1 * 100:.2f}%"
                        )

                        st.progress(
                            min(max(probability_1, 0.0), 1.0)
                        )

                        st.caption(
                            "Model output probability for Class 1."
                        )

                    else:

                        st.warning(
                            "Probability output is unavailable "
                            "for this model."
                        )

                # ---------- PROBABILITY BREAKDOWN ----------

                if probability_available:

                    st.write("")

                    prob_col1, prob_col2 = st.columns(2)

                    with prob_col1:

                        st.metric(
                            "Class 0 Probability",
                            f"{probability_0 * 100:.2f}%"
                        )

                    with prob_col2:

                        st.metric(
                            "Class 1 Probability",
                            f"{probability_1 * 100:.2f}%"
                        )

                # ---------- INPUT SUMMARY ----------

                with st.expander(
                    "View Input Parameters",
                    expanded=False
                ):

                    st.dataframe(
                        input_data,
                        use_container_width=True,
                        hide_index=True
                    )

                # ---------- DISCLAIMER ----------

                st.info(
                    "Educational machine learning demonstration only. "
                    "This model is not a medical diagnostic tool. "
                    "Predictions should not be used for medical decisions."
                )

            except Exception as error:

                st.error(
                    "Prediction error. Please verify your model "
                    "and input feature configuration."
                )

                st.exception(error)


# ==========================================================
# 8. ABOUT PROJECT PAGE
# ==========================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-title">About MediPredict</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="section-description">
        An educational Machine Learning project demonstrating
        binary classification using a Decision Tree algorithm.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="summary-card">

        <div class="summary-label">PROJECT OVERVIEW</div>

        <div class="result-text" style="margin-top:15px;">
            MediPredict is a machine learning demonstration application
            that accepts eight numerical input features and generates
            a classification result from a trained Decision Tree model.
            The interface is developed using Streamlit.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="section-title">Project Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = [
        "Load the diabetes dataset",
        "Inspect and clean the data",
        "Handle missing values",
        "Split the dataset into training and testing sets",
        "Train the Decision Tree classifier",
        "Evaluate model performance",
        "Build the Streamlit web interface",
        "Generate predictions from user inputs"
    ]

    for i, step in enumerate(workflow, start=1):

        st.markdown(f"**{i}.** {step}")

    st.write("")

    st.markdown(
        '<div class="section-title">Technology Stack</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.markdown("""
        <div class="summary-card">
            <div class="summary-value">Python</div>
            <div class="summary-label">Programming</div>
        </div>
        """, unsafe_allow_html=True)

    with tech2:

        st.markdown("""
        <div class="summary-card">
            <div class="summary-value">Scikit-learn</div>
            <div class="summary-label">Machine Learning</div>
        </div>
        """, unsafe_allow_html=True)

    with tech3:

        st.markdown("""
        <div class="summary-card">
            <div class="summary-value">Streamlit</div>
            <div class="summary-label">Web Application</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    - **Algorithm:** Decision Tree Classifier
    - **Problem Type:** Binary Classification
    - **Input:** 8 numerical features
    - **Output:** Class 0 or Class 1
    - **Dataset:** Diabetes dataset
    """)

    st.warning(
        "This project is designed for learning and demonstration. "
        "It is not clinically validated."
    )


# ==========================================================
# 9. FOOTER
# ==========================================================

st.markdown("""
<div class="footer">

    MEDIPREDICT · MACHINE LEARNING PROJECT

    <br>

    EDUCATIONAL DEMONSTRATION · NOT FOR MEDICAL USE

</div>
""", unsafe_allow_html=True)
