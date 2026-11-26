
import streamlit as st
import pandas as pd
import pickle
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="MediPredict | Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM UI / UX THEME
# =========================================================
st.markdown("""
<style>

/* ---------- Global ---------- */
html, body, [class*="stApp"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.stApp {
    background: #F4F7FA;
    color: #1E293B;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #142B3D;
    border-right: 1px solid #243F52;
}

[data-testid="stSidebar"] * {
    color: #E8F0F5 !important;
}

.sidebar-brand {
    padding: 10px 0 25px 0;
}

.sidebar-brand h1 {
    color: #FFFFFF !important;
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 4px;
}

.sidebar-brand p {
    color: #AFC4D2 !important;
    font-size: 14px;
    margin-top: 0;
}

.sidebar-divider {
    height: 1px;
    background: #2E485B;
    margin: 20px 0;
}

.sidebar-info {
    background: #1C374B;
    border: 1px solid #2E5067;
    border-radius: 12px;
    padding: 15px;
    margin-top: 15px;
}

.sidebar-info-title {
    color: #70D6C2 !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.sidebar-info-text {
    color: #D6E3EB !important;
    font-size: 14px;
    line-height: 1.8;
}

/* ---------- Main Header ---------- */
.hero {
    background: linear-gradient(135deg, #17384D 0%, #176B70 100%);
    padding: 36px 40px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(23, 56, 77, 0.10);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 38px;
    font-weight: 800;
    margin: 0;
    line-height: 1.3;
}

.hero p {
    color: #D5F0EB !important;
    font-size: 16px;
    margin: 10px 0 0 0;
}

/* ---------- Metric Cards ---------- */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E1E9F0;
    border-radius: 16px;
    padding: 22px;
    min-height: 120px;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.03);
}

.metric-label {
    color: #64748B !important;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #183247 !important;
    font-size: 25px;
    font-weight: 750;
}

.metric-sub {
    color: #8494A6 !important;
    font-size: 12px;
    margin-top: 6px;
}

/* ---------- Section Headers ---------- */
.section-title {
    color: #183247 !important;
    font-size: 23px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 5px;
}

.section-caption {
    color: #718096 !important;
    font-size: 14px;
    margin-bottom: 18px;
}

/* ---------- Form Container ---------- */
div[data-testid="stForm"] {
    background: #FFFFFF;
    padding: 24px;
    border: 1px solid #E1E9F0;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.035);
}

div[data-testid="stWidgetLabel"] p {
    color: #334155 !important;
    font-size: 14px;
    font-weight: 600;
}

/* ---------- Inputs ---------- */
div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] input {
    color: #183247 !important;
    background: #FFFFFF !important;
}

div[data-baseweb="select"] {
    background: #FFFFFF !important;
}

.stNumberInput button {
    background: #EDF4F6 !important;
    color: #176B70 !important;
}

/* ---------- Buttons ---------- */
.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(135deg, #176B70, #13877F) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 48px;
    transition: 0.2s ease;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #125C62 !important;
    border: none !important;
    transform: translateY(-1px);
}

/* ---------- Result Cards ---------- */
.result-card {
    padding: 25px;
    border-radius: 18px;
    margin-top: 12px;
    margin-bottom: 12px;
}

.result-negative {
    background: #EAF5F0;
    border: 1px solid #B9E0CC;
}

.result-positive {
    background: #FFF3E9;
    border: 1px solid #F4D0AE;
}

.result-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #526577 !important;
    margin-bottom: 10px;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
    line-height: 1.4;
}

.result-negative .result-value {
    color: #16734A !important;
}

.result-positive .result-value {
    color: #B75B21 !important;
}

.result-description {
    font-size: 14px;
    color: #526577 !important;
    margin-top: 8px;
    line-height: 1.6;
}

/* ---------- Probability Cards ---------- */
.prob-card {
    background: #FFFFFF;
    border: 1px solid #E1E9F0;
    padding: 20px;
    border-radius: 14px;
    margin-top: 10px;
}

.prob-label {
    font-size: 14px;
    color: #64748B !important;
    font-weight: 600;
}

.prob-value {
    font-size: 27px;
    font-weight: 800;
    color: #183247 !important;
    margin-top: 8px;
}

/* ---------- Notice ---------- */
.notice {
    background: #EAF1F8;
    border: 1px solid #D1DFEC;
    border-radius: 12px;
    padding: 17px 20px;
    color: #294B66 !important;
    font-size: 14px;
    line-height: 1.7;
    margin-top: 22px;
}

.footer {
    text-align: center;
    color: #8A9AAA !important;
    font-size: 12px;
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid #E1E9F0;
}

/* ---------- Tables ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid #E1E9F0;
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Hide unnecessary elements ---------- */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
    .hero {
        padding: 25px;
    }

    .hero h1 {
        font-size: 29px;
    }

    .metric-card {
        margin-bottom: 10px;
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
MODEL_PATH = "decision_tree_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# =========================================================
# FEATURES
# =========================================================
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <h1>🩺 MediPredict</h1>
        <p>Machine Learning Health Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "About Project"],
        label_visibility="visible"
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info">
        <div class="sidebar-info-title">MODEL INFORMATION</div>
        <br>
        <div class="sidebar-info-text">
            <b>Algorithm</b><br>
            Decision Tree Classifier<br><br>

            <b>Prediction Type</b><br>
            Binary Classification<br><br>

            <b>Input Features</b><br>
            8 Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info">
        <div class="sidebar-info-title">PROJECT STATUS</div>
        <br>
        <div class="sidebar-info-text">
            Machine Learning Model<br>
            Interactive Prediction Interface<br>
            Educational Demonstration
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🩺 MediPredict</h1>
    <p>Diabetes Prediction Dashboard powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL VALIDATION
# =========================================================
if model is None:

    st.error(
        "Model file not found. Please upload "
        "`decision_tree_model.pkl` into the same folder as `app.py`."
    )

    st.stop()


# Validate feature count
if hasattr(model, "n_features_in_"):

    if model.n_features_in_ != len(FEATURES):

        st.error(
            f"Your saved model expects {model.n_features_in_} input features, "
            f"but this app provides {len(FEATURES)} features. "
            "Please retrain the model without including Outcome in X "
            "and save the corrected model."
        )

        st.stop()


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    # ---------- Overview Metrics ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MODEL ALGORITHM</div>
            <div class="metric-value">Decision Tree</div>
            <div class="metric-sub">Supervised Learning</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">INPUT FEATURES</div>
            <div class="metric-value">08 Parameters</div>
            <div class="metric-sub">Health-related attributes</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">PREDICTION TYPE</div>
            <div class="metric-value">Binary</div>
            <div class="metric-sub">Classification</div>
        </div>
        """, unsafe_allow_html=True)


    # ---------- Input Form ----------
    st.markdown('<div class="section-title">Patient Input Parameters</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="section-caption">Enter values to explore the trained model output.</div>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        st.markdown("### General Parameters")

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
                step=1.0,
                help="Diastolic blood pressure"
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Skin thickness measurement"
            )

        with col2:

            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=1000.0,
                value=80.0,
                step=1.0,
                help="Insulin measurement"
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=0.1,
                help="Body Mass Index"
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
                format="%.2f",
                help="Diabetes pedigree function value"
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=25,
                step=1,
                help="Age in years"
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "✨ Generate Prediction",
            use_container_width=True
        )


    # ---------- Prediction ----------
    if submitted:

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age
            ]],
            columns=FEATURES
        )

        try:

            prediction = int(model.predict(input_data)[0])

            probabilities = model.predict_proba(input_data)[0]

            classes = list(model.classes_)

            # Probability of class 0 and class 1
            probability_0 = float(probabilities[classes.index(0)])
            probability_1 = float(probabilities[classes.index(1)])

            st.markdown("---")

            st.markdown(
                '<div class="section-title">Prediction Results</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-caption">Results generated by your trained Decision Tree model.</div>',
                unsafe_allow_html=True
            )

            # ---------- Main Result ----------
            if prediction == 0:

                st.markdown("""
                <div class="result-card result-negative">
                    <div class="result-title">MODEL CLASSIFICATION</div>
                    <div class="result-value">Class 0</div>
                    <div class="result-description">
                        The model classified the entered parameters
                        as Class 0.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-card result-positive">
                    <div class="result-title">MODEL CLASSIFICATION</div>
                    <div class="result-value">Class 1</div>
                    <div class="result-description">
                        The model classified the entered parameters
                        as Class 1.
                    </div>
                </div>
                """, unsafe_allow_html=True)


            # ---------- Probabilities ----------
            col1, col2 = st.columns(2)

            with col1:

                st.markdown(f"""
                <div class="prob-card">
                    <div class="prob-label">Class 0 Probability</div>
                    <div class="prob-value">{probability_0 * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(probability_0)

            with col2:

                st.markdown(f"""
                <div class="prob-card">
                    <div class="prob-label">Class 1 Probability</div>
                    <div class="prob-value">{probability_1 * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(probability_1)


            # ---------- Input Summary ----------
            with st.expander("View Input Parameters", expanded=False):

                input_summary = pd.DataFrame({
                    "Parameter": FEATURES,
                    "Entered Value": [
                        pregnancies,
                        glucose,
                        blood_pressure,
                        skin_thickness,
                        insulin,
                        bmi,
                        dpf,
                        age
                    ]
                })

                st.dataframe(
                    input_summary,
                    use_container_width=True,
                    hide_index=True
                )


            # ---------- Information ----------
            st.markdown("""
            <div class="notice">
                <b>Educational Demonstration:</b>
                This application demonstrates a machine learning
                classification model. The prediction and probabilities
                are outputs of the trained model and are not a medical
                diagnosis. They should not be used to make healthcare
                decisions.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:

            st.error(f"Prediction error: {e}")

            st.info(
                "Check that your saved model was trained using "
                "the same 8 feature columns in the same order."
            )

    else:

        st.markdown("""
        <div class="notice">
            Enter the parameters above and click
            <b>Generate Prediction</b> to view the model output.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================
elif page == "About Project":

    st.markdown(
        '<div class="section-title">About MediPredict</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    MediPredict is an educational machine learning project
    designed to demonstrate binary classification using a
    Decision Tree Classifier.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MACHINE LEARNING ALGORITHM</div>
            <div class="metric-value">Decision Tree Classifier</div>
            <div class="metric-sub">
                Supervised machine learning
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">PROBLEM TYPE</div>
            <div class="metric-value">Binary Classification</div>
            <div class="metric-sub">
                Two model output classes
            </div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown('<div class="section-title">Model Input Features</div>',
                unsafe_allow_html=True)

    st.markdown("""
    The model uses the following eight input features:
    """)

    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            "Number of pregnancies",
            "Glucose measurement",
            "Blood pressure measurement",
            "Skin thickness measurement",
            "Insulin measurement",
            "Body Mass Index",
            "Diabetes pedigree function",
            "Age in years"
        ]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-title">Workflow</div>',
                unsafe_allow_html=True)

    st.markdown("""
    1. User enters the eight input parameters.
    2. The input values are arranged in the correct feature order.
    3. The saved Decision Tree model processes the inputs.
    4. The model returns a predicted class.
    5. The model probabilities are displayed in the dashboard.
    """)

    st.markdown("""
    <div class="notice">
        <b>Disclaimer:</b> MediPredict is an academic machine
        learning demonstration and is not a clinical diagnostic system.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    MEDIPREDICT · MACHINE LEARNING PROJECT<br>
    Educational Demonstration Only · Not a Medical Diagnostic Tool
</div>
""", unsafe_allow_html=True)
