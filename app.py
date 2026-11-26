
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ADVANCED UI / UX CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #12243A;
    --teal: #0D9488;
    --teal-dark: #0F766E;
    --background: #F4F7FB;
    --white: #FFFFFF;
    --text: #172B4D;
    --muted: #64748B;
}

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #F4F7FB !important;
    color: #12243A !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12243A 0%, #183C55 100%) !important;
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D8E7F0 !important;
}

.sidebar-brand {
    padding: 12px 5px 20px 5px;
}

.sidebar-logo {
    font-size: 38px;
    margin-bottom: 8px;
}

.sidebar-title {
    color: #FFFFFF !important;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    color: #B8D8E7 !important;
    font-size: 13px;
}

/* ================= HERO ================= */

.hero {
    background: linear-gradient(120deg, #12243A, #164E63, #0D9488);
    padding: 38px;
    border-radius: 24px;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(15,50,75,0.12);
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.20);
    padding: 7px 12px;
    border-radius: 20px;
    color: #FFFFFF !important;
    font-size: 12px;
    margin-bottom: 15px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 35px !important;
    font-weight: 800 !important;
    margin-bottom: 10px !important;
}

.hero p {
    color: #D7F5F0 !important;
    font-size: 15px !important;
}

/* ================= METRIC CARDS ================= */

.metric-card {
    background: #FFFFFF !important;
    border: 1px solid #E4EAF1;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 4px 16px rgba(18,36,58,0.035);
    min-height: 115px;
}

.metric-label {
    color: #64748B !important;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.metric-value {
    color: #12243A !important;
    font-size: 23px;
    font-weight: 800;
    margin-top: 10px;
}

/* ================= HEADINGS ================= */

.section-heading {
    color: #12243A !important;
    font-size: 23px;
    font-weight: 800;
    margin: 10px 0 5px;
}

.section-subtitle {
    color: #64748B !important;
    font-size: 14px;
    margin-bottom: 20px;
}

h1, h2, h3, h4 {
    color: #12243A !important;
}

/* ================= INPUT LABELS ================= */

label {
    color: #243B53 !important;
    font-weight: 600 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #243B53 !important;
    font-weight: 600 !important;
}

/* ================= NUMBER INPUTS ================= */

div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] input {
    color: #12243A !important;
    background: #FFFFFF !important;
    -webkit-text-fill-color: #12243A !important;
}

div[data-baseweb="input"] svg {
    fill: #52677D !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #0D9488 !important;
    box-shadow: 0 0 0 2px rgba(13,148,136,0.12) !important;
}

/* ================= BUTTONS ================= */

div.stFormSubmitButton button,
div.stButton button {
    background: linear-gradient(90deg, #0D9488, #0F766E) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    height: 48px !important;
    font-weight: 700 !important;
    transition: 0.2s ease !important;
}

div.stFormSubmitButton button p,
div.stButton button p {
    color: #FFFFFF !important;
}

div.stFormSubmitButton button:hover,
div.stButton button:hover {
    background: #115E59 !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 18px rgba(13,148,136,0.20);
    transform: translateY(-1px);
}

/* ================= RESULT CARDS ================= */

.result-card {
    border-radius: 20px;
    padding: 26px;
    background: #FFFFFF;
    border: 1px solid #E1E8F0;
    box-shadow: 0 5px 22px rgba(18,36,58,0.04);
}

.result-positive {
    background: #FFF7F2 !important;
    border: 1px solid #FED7AA !important;
    border-left: 5px solid #EA580C !important;
}

.result-negative {
    background: #F0FDFA !important;
    border: 1px solid #99F6E4 !important;
    border-left: 5px solid #0D9488 !important;
}

.result-title {
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-description {
    color: #475569 !important;
    font-size: 14px;
    line-height: 1.7;
}

/* ================= STREAMLIT METRICS ================= */

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5ECF3 !important;
    padding: 18px !important;
    border-radius: 14px !important;
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricLabel"] * {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #12243A !important;
}

[data-testid="stMetricValue"] * {
    color: #12243A !important;
}

/* ================= EXPANDER FIX ================= */

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE5EF !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin-top: 12px !important;
}

[data-testid="stExpander"] details {
    background: #FFFFFF !important;
}

[data-testid="stExpander"] summary {
    background: #FFFFFF !important;
    color: #12243A !important;
    padding: 16px 20px !important;
    min-height: 55px !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
}

[data-testid="stExpander"] summary p {
    color: #12243A !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

[data-testid="stExpander"] summary svg {
    color: #0D9488 !important;
    fill: #0D9488 !important;
    flex-shrink: 0 !important;
}

[data-testid="stExpander"] summary:hover {
    background: #F0FDFA !important;
}

[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
    padding: 15px !important;
    border-top: 1px solid #E5EAF0 !important;
}

/* ================= CLEAN INPUT TABLE ================= */

.input-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 12px;
    overflow: hidden;
    background: #FFFFFF;
}

.input-table th {
    background: #EAF2F8;
    color: #12243A;
    font-weight: 700;
    padding: 13px;
    text-align: left;
    border: 1px solid #DCE5EF;
    font-size: 13px;
}

.input-table td {
    background: #FFFFFF;
    color: #12243A;
    padding: 13px;
    border: 1px solid #E5EAF0;
    font-size: 14px;
}

.input-table tr:nth-child(even) td {
    background: #F8FAFC;
}

/* ================= INFO / WARNING ================= */

[data-testid="stAlert"] {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 12px !important;
}

[data-testid="stAlert"] p {
    color: #1E3A8A !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

[data-testid="stAlert"] svg {
    color: #2563EB !important;
}

/* ================= PROGRESS BAR ================= */

[data-testid="stProgressBar"] > div > div {
    background-color: #0D9488 !important;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #8492A6 !important;
    font-size: 12px;
    padding: 25px 0 10px;
    line-height: 1.8;
}

/* ================= RESPONSIVE ================= */

@media (max-width: 768px) {
    .hero {
        padding: 25px;
    }

    .hero h1 {
        font-size: 27px !important;
    }

    .metric-value {
        font-size: 19px;
    }

    .block-container {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")


try:
    model = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error("Model loading failed. Please check diabetes_model.pkl.")
    st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🩺</div>
        <div class="sidebar-title">MediPredict AI</div>
        <div class="sidebar-subtitle">Intelligent Health Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "About Project"]
    )

    st.divider()

    st.markdown("### MODEL OVERVIEW")

    st.markdown("""
    **Algorithm**

    Decision Tree Classifier

    **Model Type**

    Binary Classification

    **Input Features**

    8 Parameters
    """)

    st.divider()

    if model_loaded:

        st.markdown("""
        <div style="background:rgba(255,255,255,.08);
        padding:13px;border-radius:12px;">
            <p style="font-size:12px;color:#D5E7F0 !important;">
                PROJECT STATUS
            </p>
            <p style="color:#5EEAD4 !important;font-weight:700;">
                ● Model Loaded Successfully
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.warning("Model unavailable")

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("Machine Learning Project · 2026")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">
        ✦ MACHINE LEARNING · HEALTH ANALYTICS
    </div>

    <h1>Diabetes Prediction Dashboard</h1>

    <p>
        Explore a machine learning classification demonstration
        powered by a trained Decision Tree model.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    # ---------------- METRIC CARDS ----------------

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MODEL ALGORITHM</div>
            <div class="metric-value">Decision Tree</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">INPUT FEATURES</div>
            <div class="metric-value">08 Parameters</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">CLASSIFICATION</div>
            <div class="metric-value">Binary Output</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- INPUT SECTION ----------------

    st.markdown("""
    <div class="section-heading">Patient Input Parameters</div>

    <div class="section-subtitle">
        Enter the parameters below to generate a model prediction.
    </div>
    """, unsafe_allow_html=True)

    if model_loaded:

        with st.form("prediction_form"):

            col1, col2 = st.columns(2, gap="large")

            # -------- COLUMN 1 --------

            with col1:

                st.markdown("#### 🧬 General Parameters")

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

            # -------- COLUMN 2 --------

            with col2:

                st.markdown("#### 🩺 Additional Parameters")

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

                dpf = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.50,
                    step=0.01
                )

                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=25,
                    step=1
                )

            st.markdown("<br>", unsafe_allow_html=True)

            submitted = st.form_submit_button(
                "✦  Generate Prediction",
                use_container_width=True
            )

        # ====================================================
        # PREDICTION
        # ====================================================

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

            try:

                prediction = model.predict(input_data)[0]

                probabilities = model.predict_proba(input_data)[0]

                class_0_probability = float(probabilities[0])
                class_1_probability = float(probabilities[1])

                # ---------------- RESULTS ----------------

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("""
                <div class="section-heading">Prediction Analysis</div>

                <div class="section-subtitle">
                    Model output based on the entered parameters.
                </div>
                """, unsafe_allow_html=True)

                result_col, probability_col = st.columns(
                    [1.2, 1],
                    gap="large"
                )

                # -------- RESULT CARD --------

                with result_col:

                    if prediction == 1:

                        st.markdown("""
                        <div class="result-card result-positive">

                            <div style="font-size:30px;">📊</div>

                            <div class="result-title"
                            style="color:#9A3412 !important;">
                                Model Output: Class 1
                            </div>

                            <p class="result-description">
                                The model classified this input into Class 1.
                                This is an educational prediction and is
                                not a medical diagnosis.
                            </p>

                        </div>
                        """, unsafe_allow_html=True)

                    else:

                        st.markdown("""
                        <div class="result-card result-negative">

                            <div style="font-size:30px;">📊</div>

                            <div class="result-title"
                            style="color:#115E59 !important;">
                                Model Output: Class 0
                            </div>

                            <p class="result-description">
                                The model classified this input into Class 0.
                                This is an educational prediction and is
                                not a medical diagnosis.
                            </p>

                        </div>
                        """, unsafe_allow_html=True)

                # -------- PROBABILITY CARD --------

                with probability_col:

                    st.markdown("""
                    <div class="result-card">
                        <div style="font-weight:700;
                        font-size:17px;color:#12243A !important;">
                            Model Probability
                        </div>

                        <p style="color:#64748B !important;
                        font-size:13px;">
                            Probability assigned to Class 1
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric(
                        "Class 1 Probability",
                        f"{class_1_probability * 100:.2f}%"
                    )

                    st.progress(class_1_probability)

                    st.caption(
                        "Model-estimated probability, not actual medical risk."
                    )

                # -------- PROBABILITY METRICS --------

                st.markdown("<br>", unsafe_allow_html=True)

                p1, p2 = st.columns(2)

                with p1:

                    st.metric(
                        "Class 0 Probability",
                        f"{class_0_probability * 100:.2f}%"
                    )

                with p2:

                    st.metric(
                        "Class 1 Probability",
                        f"{class_1_probability * 100:.2f}%"
                    )

                # -------- INPUT DATA TABLE --------

                with st.expander(
                    "📋 View Input Parameters",
                    expanded=False
                ):

                    st.markdown(
                        input_data.to_html(
                            index=False,
                            classes="input-table",
                            border=0
                        ),
                        unsafe_allow_html=True
                    )

                # -------- DISCLAIMER --------

                st.info(
                    "Educational ML demonstration only. "
                    "Do not use this output for diagnosis or treatment decisions."
                )

            except Exception as e:

                st.error("Prediction failed. Please check your input values.")
                st.exception(e)


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.markdown("""
    <div class="result-card">

        <h2 style="color:#12243A !important;">
            About MediPredict AI
        </h2>

        <p style="color:#475569 !important;line-height:1.8;">
            MediPredict is an educational Machine Learning project
            demonstrating binary classification using a Decision
            Tree model trained with a diabetes dataset.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">Project Workflow</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    1. Dataset loading using Pandas
    2. Data cleaning and preprocessing
    3. Missing value handling
    4. Train/test split
    5. Decision Tree model training
    6. Prediction and evaluation
    7. Web application deployment
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">Technology Stack</div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)

    with t1:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Python</div>
            <div class="metric-label">Programming Language</div>
        </div>
        """, unsafe_allow_html=True)

    with t2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Scikit-learn</div>
            <div class="metric-label">Machine Learning</div>
        </div>
        """, unsafe_allow_html=True)

    with t3:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Streamlit</div>
            <div class="metric-label">Web Framework</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    MEDIPREDICT AI · HEALTH ANALYTICS DASHBOARD

    <br><br>

    EDUCATIONAL PROJECT · NOT A MEDICAL DIAGNOSTIC TOOL

</div>
""", unsafe_allow_html=True)
