import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")

# -------------------------------
# Load All Models (with pipeline)
# -------------------------------
@st.cache_resource
def load_models():
    models = {
        "Linear Regression": joblib.load("models/best_model_Linear.joblib"),
        "Ridge Regression": joblib.load("models/best_model_Ridge.joblib"),
        "Lasso Regression": joblib.load("models/best_model_Lasso.joblib"),
        "Random Forest": joblib.load("models/best_model_RandomForest.joblib"),
        "Gradient Boosting": joblib.load("models/best_model_GradientBoosting.joblib"),
        "XGBoost": joblib.load("models/best_model_XGBoost.joblib"),
        "SVR": joblib.load("models/best_model_SVR.joblib"),
        "LightGBM": joblib.load("models/best_model_LightGBM.joblib")
    }
    return models

models = load_models()
model_names = list(models.keys())

# -------------------------------
# App Title
# -------------------------------
st.title("💰 Insurance Charges Prediction App")
st.markdown("This app uses several regression models to estimate insurance charges based on customer inputs.")

# -------------------------------
# Model Selection
# -------------------------------
model_selection = st.selectbox("Select a Model", ["All Models"] + model_names)

# -------------------------------
# Input Form
# -------------------------------
st.subheader("Enter Customer Details")

with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])
    submitted = st.form_submit_button("Predict")

# -------------------------------
# Predict
# -------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    st.subheader("📊 Prediction Result")

    if model_selection == "All Models":
        results = []
        for name, model in models.items():
            try:
                prediction = model.predict(input_df)[0]
                results.append({"Model": name, "Predicted Charges (₹)": round(prediction, 2)})
            except Exception as e:
                results.append({"Model": name, "Predicted Charges (₹)": f"Error: {e}"})

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.format({"Predicted Charges (₹)": "₹{:.2f}"}), use_container_width=True)

    else:
        # Predict using selected model
        model = models[model_selection]
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💸 **{model_selection} Prediction:** ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction error with {model_selection}: {e}")
