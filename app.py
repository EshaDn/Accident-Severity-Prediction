import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Road Safety | Severity Assessment",
    page_icon="🚘",
    layout="wide"
)

# -------------------- CUSTOM DESIGN --------------------

st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #263746;
}

[data-testid="stHeader"] {
    background-color: #FFFFFF;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #263746 !important;
    font-weight: 600 !important;
}

p, label {
    color: #596B78;
}

/* Model selector visibility */
[data-testid="stSelectbox"] {
    overflow: visible !important;
    margin-bottom: 1rem;
}

[data-testid="stWidgetLabel"] {
    color: #263746 !important;
    opacity: 1 !important;
    padding-bottom: 0.4rem !important;
    line-height: 1.5 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #263746 !important;
    opacity: 1 !important;
    line-height: 1.5 !important;
    font-weight: 600 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #263746 !important;
    border: 1px solid #CEDAE3 !important;
    border-radius: 6px;
    min-height: 48px !important;
    align-items: center !important;
}

.stSelectbox div[data-baseweb="select"] span {
    color: #263746 !important;
    opacity: 1 !important;
}

[data-testid="stForm"] {
    background-color: #F7FAFC;
    padding: 26px;
    border: 1px solid #DCE6ED;
    border-radius: 10px;
}

.stNumberInput input {
    background-color: #FFFFFF;
    color: #263746;
    border: 1px solid #CEDAE3;
    border-radius: 6px;
}

.stFormSubmitButton button {
    background-color: #3D647D !important;
    color: #FFFFFF !important;
    border: 1px solid #3D647D;
    border-radius: 7px;
    min-height: 46px;
    font-weight: 600;
}

.stFormSubmitButton button p {
    color: #FFFFFF !important;
}

[data-testid="stAlert"] {
    border-radius: 8px;
}

hr {
    border-color: #DCE6ED;
}

div[data-testid="stMetric"] {
    background-color: #EAF1F7;
    padding: 16px;
    border: 1px solid #DCE6ED;
    border-radius: 9px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- MODEL SELECTION --------------------

MODEL_FILES = {
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Improved Random Forest": "improved_random_forest.pkl"
}


# -------------------- DASHBOARD HEADER --------------------

st.title("Road Accident Severity Assessment")

st.markdown(
    "An analytical tool for estimating road accident severity "
    "using machine learning and Indian highway accident records."
)

# Single model selector
selected_model_name = st.selectbox(
    "Select Prediction Model",
    list(MODEL_FILES.keys()),
    key="prediction_model_selector"
)

# Load selected model
MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / MODEL_FILES[selected_model_name]
)

model = joblib.load(MODEL_PATH)

st.write("")


# -------------------- FEATURE INFORMATION --------------------

categorical_features = [
    'Day_of_Week',
    'Accident_Location_A',
    'Accident_Location_A_Chainage_km_RoadSide',
    'Nature_of_Accident_B1',
    'Road_Feature_E',
    'Road_Condition_F',
    'Intersection_Type_G',
    'Weather_Conditions_H',
    'Vehicle_Type_Involved_J_V1',
    'Number_of_Vehicles'
]

numerical_features = [
    'Accident_Hour',
    'Accident_Year',
    'Accident_Month'
]


# -------------------- GET VALID CATEGORIES --------------------

encoder = (
    model.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['encoder']
)

category_options = {
    feature: list(categories)
    for feature, categories in zip(
        categorical_features,
        encoder.categories_
    )
}


# -------------------- DASHBOARD OVERVIEW --------------------

st.markdown("### Model Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Dataset Records",
        value="3,253"
    )

with col2:
    st.metric(
        label="Severity Categories",
        value="04"
    )

st.write("")
st.divider()


# -------------------- USER-FRIENDLY LABELS --------------------

def code_key(value):
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (ValueError, TypeError):
        pass

    return str(value)


def format_option(value, mapping):
    key = code_key(value)
    return mapping.get(
        key,
        f"Undocumented category ({key})"
    )


location_labels = {
    "1": "Urban",
    "2": "Rural"
}

nature_labels = {
    "1": "Overturning",
    "2": "Head-on Collision",
    "3": "Rear-end Collision",
    "4": "Brush / Side-wipe Collision",
    "5": "Right-turn Collision",
    "6": "Skidding",
    "7a": "Hit Cyclist",
    "7b": "Hit Pedestrian",
    "7c": "Hit Parked Vehicle",
    "7d": "Hit Fixed Object",
    "7e": "Wrong-side Driving",
    "7f": "Hit Animal",
    "7g": "Hit Two-wheeler",
    "7h": "Other / Unknown",
    "8": "Overtaking Vehicle",
    "9": "Left-turn Collision"
}

road_feature_labels = {
    "1": "Single Lane",
    "2": "Two Lanes",
    "3": "Three or More Lanes (No Central Divider)",
    "4": "Four or More Lanes (With Central Divider)"
}

road_condition_labels = {
    "1": "Straight Road",
    "2": "Slight Curve",
    "3": "Sharp Curve",
    "4": "Flat Road",
    "5": "Gentle Incline",
    "6": "Steep Incline",
    "7": "Hump",
    "8": "Dip"
}

intersection_labels = {
    "1": "T-Junction",
    "2": "Y-Junction",
    "3": "Four-arm Junction",
    "4": "Staggered Junction",
    "5": "Roundabout",
    "6": "Uncontrolled Junction"
}

weather_labels = {
    "1": "Fine",
    "2": "Mist / Fog",
    "3": "Cloud",
    "4": "Light Rain",
    "5": "Heavy Rain",
    "6": "Hail / Sleet",
    "7": "Snow",
    "8": "Strong Wind",
    "9": "Dust Storm",
    "10": "Very Hot",
    "11": "Very Cold",
    "12": "Other Extraordinary Weather"
}

vehicle_labels = {
    "1": "Car / Jeep / Van",
    "2": "SUV",
    "3": "Bus",
    "4": "Mini Bus",
    "5": "Truck",
    "6": "Two Wheeler",
    "7": "Three Wheeler",
    "8": "Cycle",
    "9": "Pedestrian",
    "10": "Tractor",
    "11": "Unknown",
    "12": "Animal",
    "14": "LCV",
    "15": "MAV"
}

day_mapping = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday"
}


# -------------------- ACCIDENT INPUT FORM --------------------

st.subheader("Accident Assessment")
st.caption("Enter the recorded accident information.")

with st.form("prediction_form"):

    st.markdown("**01  |  Time & Location**")

    col1, col2 = st.columns(2)

    with col1:
        day = st.selectbox(
            "Day of Week",
            options=list(day_mapping.keys()),
            format_func=lambda x: day_mapping[x]
        )

        year = st.selectbox(
            "Accident Year",
            [2013, 2014, 2015, 2016, 2017, 2018]
        )

        location = st.selectbox(
            "Accident Location",
            category_options['Accident_Location_A'],
            format_func=lambda x: format_option(
                x, location_labels
            )
        )

    with col2:
        hour = st.select_slider(
            "Accident Time",
            options=list(range(24)),
            value=12,
            format_func=lambda x: (
                f"{x % 12 or 12}:00 "
                f"{'AM' if x < 12 else 'PM'}"
            )
        )

        month = st.slider(
            "Accident Month",
            1, 12, 6
        )

        roadside = st.selectbox(
            "Roadside",
            category_options[
                'Accident_Location_A_Chainage_km_RoadSide'
            ],
            format_func=lambda x: {
                "LHS": "Left-hand side",
                "RHS": "Right-hand side"
            }.get(str(x), str(x))
        )

    st.divider()

    st.markdown("**02  |  Road & Environment**")

    col1, col2 = st.columns(2)

    with col1:
        road_feature = st.selectbox(
            "Road Feature",
            category_options['Road_Feature_E'],
            format_func=lambda x: format_option(
                x, road_feature_labels
            )
        )

        road_condition = st.selectbox(
            "Road Condition",
            category_options['Road_Condition_F'],
            format_func=lambda x: format_option(
                x, road_condition_labels
            )
        )

    with col2:
        intersection = st.selectbox(
            "Intersection Type",
            category_options['Intersection_Type_G'],
            format_func=lambda x: format_option(
                x, intersection_labels
            )
        )

        weather = st.selectbox(
            "Weather Conditions",
            category_options['Weather_Conditions_H'],
            format_func=lambda x: format_option(
                x, weather_labels
            )
        )

    st.divider()

    st.markdown("**03  |  Accident Characteristics**")

    col1, col2 = st.columns(2)

    with col1:
        nature = st.selectbox(
            "Nature of Accident",
            category_options['Nature_of_Accident_B1'],
            format_func=lambda x: format_option(
                x, nature_labels
            )
        )

    with col2:
        vehicle = st.selectbox(
            "Vehicle Type",
            category_options['Vehicle_Type_Involved_J_V1'],
            format_func=lambda x: format_option(
                x, vehicle_labels
            )
        )

    number_vehicles = st.selectbox(
        "Number of Vehicles",
        category_options['Number_of_Vehicles'],
        format_func=lambda x: (
            f"{code_key(x)} vehicle"
            if code_key(x) == "1"
            else f"{code_key(x)} vehicles"
        )
    )

    st.write("")

    submitted = st.form_submit_button(
        "Predict Accident Severity",
        use_container_width=True
    )


# -------------------- PREDICTION --------------------

# -------------------- PREDICTION --------------------

if submitted:

    input_data = pd.DataFrame([{
        'Day_of_Week': str(day),
        'Accident_Hour': hour,
        'Accident_Year': year,
        'Accident_Month': month,
        'Accident_Location_A': str(location),
        'Accident_Location_A_Chainage_km_RoadSide': str(roadside),
        'Nature_of_Accident_B1': str(nature),
        'Road_Feature_E': str(road_feature),
        'Road_Condition_F': str(road_condition),
        'Intersection_Type_G': str(intersection),
        'Weather_Conditions_H': str(weather),
        'Vehicle_Type_Involved_J_V1': str(vehicle),
        'Number_of_Vehicles': str(number_vehicles)
    }])

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    severity_labels = {
        1: "Fatal",
        2: "Grievous Injury",
        3: "Minor Injury",
        4: "No Injury"
    }

    severity_colors = {
        1: "#C75B5B",
        2: "#D18B43",
        3: "#3D7896",
        4: "#43866C"
    }

    result_color = severity_colors[prediction]

    st.divider()

    st.subheader("Prediction Assessment")
    st.caption("MODEL-GENERATED RESULT")

    # Beautiful prediction card
    with st.container(border=True):

        st.markdown(
            "<p style='color:#596B78; font-size:12px; "
            "font-weight:600; letter-spacing:1px; "
            "margin-bottom:8px;'>"
            "PREDICTED ACCIDENT SEVERITY</p>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h2 style='color:{result_color}; "
            f"font-size:30px; margin-top:0;'>"
            f"{severity_labels[prediction]}</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='height:4px; width:65px; "
            f"background-color:{result_color}; "
            f"border-radius:4px;'></div>",
            unsafe_allow_html=True
        )

    st.write("")

    # Severity score distribution
    st.subheader("Severity Score Distribution")

    score_data = pd.DataFrame({
        "Severity": [
            severity_labels[int(label)]
            for label in model.named_steps['classifier'].classes_
        ],
        "Model Score (%)": probabilities * 100
    })

    st.bar_chart(
        score_data.set_index("Severity"),
        color="#7796A9"
    )

    st.caption(
        "Scores represent model-estimated probabilities, "
        "not guaranteed real-world probabilities."
    )
# -------------------- PROJECT INFORMATION --------------------

st.divider()

st.subheader("Project Information")
st.caption("MODEL & DATASET DETAILS")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Dataset**")
    st.write("Indian Highway Accident Records")

    st.markdown("**Prediction Categories**")
    st.write("4 Severity Classes")

with col2:
    st.markdown(
        "**Improved Random Forest — 5-Fold CV Macro F1**"
    )
    st.write("0.3827 ± 0.0115")

st.caption(
    "Academic Machine Learning Project | Road Accident Severity Prediction"
)