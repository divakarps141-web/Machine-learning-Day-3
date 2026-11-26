
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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
# PROFESSIONAL UI STYLING
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --background: #f5f7fb;
    --surface: #ffffff;
    --primary: #0f766e;
    --primary-light: #e6f5f2;
    --text: #172b4d;
    --muted: #64748b;
    --border: #e2e8f0;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: var(--background);
}

/* Main content */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #14263f;
    border-right: 1px solid #263a54;
}

section[data-testid="stSidebar"] * {
    color: #eaf1fa !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #eaf1fa !important;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #172b4d !important;
    font-weight: 700 !important;
}

p, label, span {
    font-family: 'Inter', sans-serif;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #14304d 0%, #0f766e 100%);
    padding: 36px 40px;
    border-radius: 22px;
    margin-bottom: 26px;
    border: 1px solid rgba(255,255,255,0.10);
}

.hero-title {
    color: #ffffff !important;
    font-size: 34px;
    font-weight: 800;
    margin: 0 0 10px 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #d9f0ec !important;
    font-size: 15px;
    line-height: 1.7;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e3eaf2;
    border-radius: 16px;
    padding: 22px 24px;
    min-height: 120px;
    box-shadow: 0 3px 12px rgba(20, 38, 63, 0.03);
}

.metric-label {
    color: #64748b !important;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #172b4d !important;
    font-size: 27px;
    font-weight: 750;
    line-height: 1.3;
}

.metric-description {
    color: #64748b !important;
    font-size: 12px;
    margin-top: 7px;
}

/* Section headings */
.section-heading {
    color: #172b4d !important;
    font-size: 21px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 6px;
}

.section-caption {
    color: #64748b !important;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Form container */
.form-card {
    background: #ffffff;
    border: 1px solid #e3eaf2;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 4px 16px rgba(20, 38, 63, 0.03);
}

/* Input labels */
div[data-testid="stWidgetLabel"] p {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* Inputs */
div[data-baseweb="input"] {
    background: #ffffff !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] input {
    color: #172b4d !important;
    background: #ffffff !important;
}

/* Number input buttons */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    color: #334155 !important;
}

/* Primary buttons */
.stButton button[kind="primary"],
.stFormSubmitButton button[kind="primary"] {
    background: #0f766e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 48px;
    font-weight: 700 !important;
    transition: 0.2s ease;
}

.stButton button[kind="primary"]:hover,
.stFormSubmitButton button[kind="primary"]:hover {
    background: #115e59 !important;
    border: none !important;
}

/* Secondary buttons */
.stButton button {
    border-radius: 10px !important;
}

/* Result panel */
.result-card {
    background: #ffffff;
    border: 1px solid #e3eaf2;
    border-radius: 18px;
    padding: 28px;
    margin-top: 12px;
}

.result-safe {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 14px;
    padding: 20px;
}

.result-warning {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 14px;
    padding: 20px;
}

.result-title {
    color: #172b4d !important;
    font-size: 23px;
    font-weight: 750;
    margin-bottom: 8px;
}

.result-description {
    color: #475569 !important;
    font-size: 14px;
    line-height: 1.7;
}

/* Disclaimer */
.disclaimer {
    background: #eef2f8;
    border: 1px solid #dce5ef;
    padding: 16px 20px;
    border-radius: 12px;
    color: #334155 !important;
    font-size: 13px;
    line-height: 1.7;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 12px;
    padding: 24px 0 8px 0;
}

/* Hide Streamlit menu/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 24px;
    }

    .hero-title {
        font-size: 26px;
    }

    .metric-card {
        margin-bottom: 10px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATASET CONFIGURATION
# =========================================================

DATASET_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

FEATURES = DATASET_COLUMNS[:-1]
TARGET = "Outcome"

INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset(file):

    df = pd.read_csv(file)

    # Handle datasets without column headers
    if len(df.columns) == 9 and not set(DATASET_COLUMNS).issubset(df.columns):
        df.columns = DATASET_COLUMNS

    # Validate required columns
    missing_columns = [
        column for column in DATASET_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: " + ", ".join(missing_columns)
        )

    df = df[DATASET_COLUMNS].copy()

    # Convert columns to numeric
    for column in DATASET_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace medically invalid zeros with missing values
    for column in INVALID_ZERO_COLUMNS:
        df[column] = df[column].replace(0, np.nan)

    # Remove rows with missing target
    df = df.dropna(subset=[TARGET])

    # Validate target classes
    df = df[df[TARGET].isin([0, 1])]

    return df


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model(df):

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, len(X_train), len(X_test)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 5px 25px 5px;">
        <div style="font-size:38px;">🩺</div>
        <h2 style="color:#ffffff !important; margin:8px 0;">
            MediPredict
        </h2>
        <p style="color:#b9c9dc !important; font-size:14px;">
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

    st.markdown("**Dataset**")
    st.caption("Pima Indians Diabetes Dataset")

    st.divider()

    st.markdown("""
    <div style="font-size:12px; color:#b9c9dc;">
        Student Machine Learning Project<br>
        Educational Demonstration
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# DATASET SOURCE
# =========================================================

dataset_file = None

try:
    dataset_file = open("diabetes.csv", "rb")
except FileNotFoundError:
    dataset_file = None


if dataset_file is None:

    with st.sidebar:
        st.markdown("### Dataset")

        uploaded_file = st.file_uploader(
            "Upload diabetes.csv",
            type=["csv"]
        )

    if uploaded_file is not None:
        dataset_file = uploaded_file


# =========================================================
# CHECK DATASET
# =========================================================

if dataset_file is None:

    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">MediPredict</h1>
        <p class="hero-subtitle">
            Machine Learning Diabetes Prediction Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "Dataset not found. Place diabetes.csv in the same folder "
        "as app.py or upload it using the sidebar."
    )

    st.info(
        "For GitHub deployment, upload both app.py and diabetes.csv "
        "to your repository."
    )

    st.stop()


# =========================================================
# PROCESS DATA AND TRAIN MODEL
# =========================================================

try:

    diabetes = load_dataset(dataset_file)

    model, model_accuracy, train_rows, test_rows = train_model(diabetes)

except Exception as error:

    st.error(f"Dataset error: {error}")
    st.stop()


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    # Hero section
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Diabetes Prediction Dashboard</h1>
        <p class="hero-subtitle">
            Machine learning classification demonstration using
            a Decision Tree model. Enter patient parameters to
            explore the model's prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # =====================================================
    # METRIC CARDS
    # =====================================================

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MODEL</div>
            <div class="metric-value">Decision Tree</div>
            <div class="metric-description">Classification algorithm</div>
        </div>
        """, unsafe_allow_html=True)

    with metric2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">INPUT FEATURES</div>
            <div class="metric-value">08</div>
            <div class="metric-description">Predictor parameters</div>
        </div>
        """, unsafe_allow_html=True)

    with metric3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">DATASET SIZE</div>
            <div class="metric-value">{len(diabetes):,}</div>
            <div class="metric-description">Records used</div>
        </div>
        """, unsafe_allow_html=True)

    with metric4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TEST ACCURACY</div>
            <div class="metric-value">{model_accuracy:.1%}</div>
            <div class="metric-description">Held-out test data</div>
        </div>
        """, unsafe_allow_html=True)


    st.write("")

    st.markdown(
        '<div class="section-heading">Patient Parameters</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Enter the values below to generate a model prediction.'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # INPUT FORM
    # =====================================================

    with st.form("diabetes_prediction_form"):

        st.markdown("### General Parameters")

        col1, col2 = st.columns(2, gap="large")

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
                min_value=1.0,
                max_value=300.0,
                value=120.0,
                step=1.0,
                help="Glucose concentration"
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=1.0,
                max_value=200.0,
                value=80.0,
                step=1.0,
                help="Diastolic blood pressure"
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Triceps skin fold thickness"
            )

        with col2:

            insulin = st.number_input(
                "Insulin",
                min_value=1.0,
                max_value=1000.0,
                value=120.0,
                step=1.0,
                help="2-hour serum insulin"
            )

            bmi = st.number_input(
                "BMI",
                min_value=1.0,
                max_value=80.0,
                value=25.0,
                step=0.1,
                help="Body mass index"
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
                help="Diabetes pedigree function"
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=25,
                step=1,
                help="Age in years"
            )


        st.write("")

        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary",
            use_container_width=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

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

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]

        class_0_probability = float(probabilities[0]) * 100
        class_1_probability = float(probabilities[1]) * 100


        st.write("")

        st.markdown(
            '<div class="section-heading">Prediction Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-caption">'
            'Output generated from the trained machine learning model.'
            '</div>',
            unsafe_allow_html=True
        )


        # Prediction result
        if prediction == 0:

            st.markdown("""
            <div class="result-safe">
                <div class="result-title">
                    Model Output: Class 0
                </div>
                <div class="result-description">
                    The model classified the supplied parameters
                    into Class 0.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-warning">
                <div class="result-title">
                    Model Output: Class 1
                </div>
                <div class="result-description">
                    The model classified the supplied parameters
                    into Class 1.
                </div>
            </div>
            """, unsafe_allow_html=True)


        st.write("")

        # Probability cards
        probability_col1, probability_col2 = st.columns(2, gap="large")

        with probability_col1:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CLASS 0 PROBABILITY</div>
                <div class="metric-value">{class_0_probability:.2f}%</div>
                <div class="metric-description">
                    Model probability for Class 0
                </div>
            </div>
            """, unsafe_allow_html=True)

        with probability_col2:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CLASS 1 PROBABILITY</div>
                <div class="metric-value">{class_1_probability:.2f}%</div>
                <div class="metric-description">
                    Model probability for Class 1
                </div>
            </div>
            """, unsafe_allow_html=True)


        # Probability visualization
        st.write("")

        st.markdown("### Prediction Probability")

        st.progress(
            int(round(class_1_probability)),
            text=f"Class 1: {class_1_probability:.2f}%"
        )

        st.caption(
            "These values represent the Decision Tree model's "
            "probability estimates, not actual medical risk."
        )


        # Input summary
        with st.expander("View Input Parameters"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


        st.markdown("""
        <div class="disclaimer">
            <strong>Educational Disclaimer:</strong>
            This application is an academic machine learning
            demonstration. It is not a medical diagnostic tool.
            Its predictions and probability estimates should not
            be used to diagnose diabetes or make medical decisions.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">About MediPredict</h1>
        <p class="hero-subtitle">
            A student machine learning project demonstrating
            binary classification using a Decision Tree.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("## Project Overview")

    st.write("""
    MediPredict is a machine learning demonstration that
    uses selected health-related input parameters to classify
    a dataset record into one of two output classes.

    The model is trained using the Diabetes dataset and
    demonstrates a complete machine learning workflow:
    dataset preprocessing, missing-value handling,
    training, testing, and prediction.
    """)


    st.markdown("## Machine Learning Workflow")

    workflow_col1, workflow_col2 = st.columns(2, gap="large")

    with workflow_col1:

        st.markdown("""
        ### Data Processing

        - Dataset loading
        - Numerical data validation
        - Invalid zero-value handling
        - Missing-value imputation
        - Feature and target separation
        - Stratified train-test split
        """)

    with workflow_col2:

        st.markdown("""
        ### Model Development

        - Decision Tree Classifier
        - Maximum depth: 4
        - Balanced class weights
        - Training and testing
        - Classification prediction
        - Probability output
        """)


    st.markdown("## Model Performance")

    performance_col1, performance_col2, performance_col3 = st.columns(3)

    with performance_col1:

        st.metric(
            "Test Accuracy",
            f"{model_accuracy:.2%}"
        )

    with performance_col2:

        st.metric(
            "Training Records",
            f"{train_rows:,}"
        )

    with performance_col3:

        st.metric(
            "Testing Records",
            f"{test_rows:,}"
        )


    st.markdown("## Dataset Overview")

    st.dataframe(
        diabetes.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing the first 10 records after preprocessing."
    )


    st.markdown("""
    <div class="disclaimer">
        This project is created for educational and academic
        demonstration purposes only. It is not intended for
        real-world medical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    MEDIPREDICT · MACHINE LEARNING PROJECT · EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
