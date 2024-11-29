import streamlit as st
import joblib
import pandas as pd
import numpy as np
from static.helper import continuous_log_reg_coefficients 

# Load the model
@st.cache_resource
def load_nrf_cat_model():
    return joblib.load("./static/cat_nrf_model.joblib")

@st.cache_resource
def load_nrf_cont_model():
    return joblib.load("./static/cont_nrf_model.joblib")

nrf_cat_model = load_nrf_cat_model()
nrf_cont_model = load_nrf_cont_model()

# Sidebar for Model Selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest Model", "Logistic Regression Model"])

# Page title that reflects the selected model
st.title(f"Patient Details - {model_choice}")

if model_choice == "Random Forest Model":
    # Create three columns for the sections
    col1, spacer1, col2, spacer2, col3 = st.columns([2, 0.5, 2, 0.5, 2])

    # Section 1: Medical Conditions (Categorical/Binary Features)
    with col1:
        st.header("Conditions")
        atrial_fibrillation = st.checkbox("Atrial Fibrillation")
        cva = st.checkbox("Cerebrovascular Accident (CVA)") 
        heart_failure = st.checkbox('Chronic Heart Failure')
        dementia = st.checkbox("Dementia")
        mi_nstemi = st.checkbox("Myocardial Infraction")
        pvd = st.checkbox("Peripheral Vascular Disease (PVD)")

    # Section 2: Age & Laboratory Data
    with col2:
        st.header("Lab Data")

        # Albumin level
        albumin = st.number_input("Albumin (g/L)", min_value=0.0, step=0.1, format="%.2f")
        albumin_35_abv = int(albumin >= 35)

        # eGFR level
        egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, step=0.1, format="%.2f")
        egfr_15_abv = int(egfr >= 15)
        
        # Hemoglobin level
        haemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1, format="%.2f")
        haemoglobin_10_abv = int(haemoglobin >= 10)
        
        # Phosphate Inorganic level
        phosphate = st.number_input("Phosphate Inorganic, serum (mmol/L)", min_value=0.0, step=0.1, format="%.2f")
        phosphate_1_6_abv = int(phosphate >= 1.6)

    # Section 3: Renal Function & Comorbidity Index
    with col3:
        st.header("Others")
        
        adl_dependent = st.checkbox("ADL Dependent")

        # Age categories
        age = st.number_input("Age", min_value=0, step=1)
        age_75_79 = int(75 <= age <= 79)
        age_80_84 = int(80 <= age <= 84)

        # CCI score
        cci = st.number_input("Charlson Comorbidity Index (CCI)", min_value=0, step=1)
        cci_abv_5 = int(cci > 5)

elif model_choice == "Logistic Regression Model":
    # Create three columns for the sections
    col1, spacer1, col2, spacer2, col3 = st.columns([2, 0.5, 2, 0.5, 2])

    # Section 1: Medical Conditions (Categorical/Binary Features)
    with col1:
        st.header("Conditions")
        atrial_fibrillation = st.checkbox("Atrial Fibrillation")
        cva = st.checkbox("Cerebrovascular Accident (CVA)") 
        heart_failure = st.checkbox('Chronic Heart Failure')
        dementia = st.checkbox("Dementia")
        liverdisease = st.checkbox('Liver Disease')
        mi_nstemi = st.checkbox("Myocardial Infraction")
        pvd = st.checkbox("Peripheral Vascular Disease (PVD)")

    # Section 2: Age & Laboratory Data
    with col2:
        st.header("Lab Data")

        # Albumin level
        albumin = st.number_input("Albumin (g/L)", min_value=0.0, step=0.1, format="%.2f")
        albumin_24_less = int(albumin <= 24)
        albumin_25_29 = int(25 <= albumin <= 29)
        albumin_30_34 = int(30 <= albumin <= 34)
        albumin_35_abv = int(albumin >= 35)

        # calcium level
        calcium = st.number_input("Calcium (mmol/L)", min_value=0.0, step=0.1, format="%.2f")

        # eGFR level
        egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, step=0.1, format="%.2f")
        egfr_10_14 = int(10 <= egfr <= 14)
        egfr_15_abv = int(egfr >= 15)
        
        # Hemoglobin level
        haemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1, format="%.2f")
        haemoglobin_10_abv = int(haemoglobin >= 10)
        
        # Phosphate Inorganic level
        phosphate = st.number_input("Phosphate Inorganic, serum (mmol/L)", min_value=0.0, step=0.1, format="%.2f")
        phosphate_1_6_abv = int(phosphate >= 1.6)


    # Section 3: Renal Function & Comorbidity Index
    with col3:
        st.header("Others")
        
        adl_dependent = st.checkbox("ADL Dependent")

        # Age categories
        age = st.number_input("Age", min_value=0, step=1)
        age_65_69 = int(65 <= age <= 69)
        age_70_74 = int(70 <= age <= 74)
        age_75_79 = int(75 <= age <= 79)
        age_80_84 = int(80 <= age <= 84)

        # CCI score
        cci = st.number_input("Charlson Comorbidity Index (CCI)", min_value=0, step=1)
        cci_3_4 = int(3 <= cci <= 4)
        cci_abv_5 = int(cci > 5)

        # CRRT given
        crrt = st.checkbox('CRRT Given')

        # Gender
        gender = st.selectbox("Gender", ["Male", "Female"])
        male = int(gender == 'Male')

        # race
        race = st.selectbox("Race", ["Chinese","Indian", "Malay", "Others"])
        raceIndian = int(race == 'Indian')
        raceMalay = int(race == 'Malay')
        raceOthers = int(race == 'Others')


_, center_col, _ = st.columns([1, 1, 1])  # Adjust widths if needed
# Create a placeholder for the output
output_placeholder = st.empty()

with center_col:
# Display model-specific options or calculations after submission
# Submit button
    if st.button("Submit Data"):

        if model_choice == "Random Forest Model":

            # Prepare the input data for the model as a dictionary
            input_data_categorical = {
                'Atrial fibrillation': int(atrial_fibrillation),
                'MI or NSTEMI': int(mi_nstemi),
                'PVD': int(pvd),
                'CVA': int(cva),    
                'Dementia': int(dementia),
                'ADL Dependent': int(adl_dependent),
                'age_75-79': age_75_79,
                'age_80-84': age_80_84,
                'albumin_35 abv': albumin_35_abv,
                'Haemoglobin >= 10': haemoglobin_10_abv,
                'Phosphate Inorganic, serum >= 1.6': phosphate_1_6_abv,
                'eGFR (CKD-EPI)_15 abv': egfr_15_abv,
                'cci_abv 5': cci_abv_5
            }

            rf_input_categorical_df = pd.DataFrame([input_data_categorical])

            input_data_continuous = {
                'Age' : age,
                'Atrial fibrillation': int(atrial_fibrillation),
                'MI or NSTEMI': int(mi_nstemi),
                'CVA': int(cva),    
                'Chronic Heart Failure (merged)' : int(heart_failure),
                'Albumin, serum' : albumin,
                'Haemoglobin' : haemoglobin,
                'Phosphate Inorganic, serum' : phosphate,
                'eGFR (CKD-EPI)': egfr,
                'ADL Dependent' : int(adl_dependent),
                'cci' : cci
            }

            rf_input_continuous_df = pd.DataFrame([input_data_continuous])

            # Insert model-specific logic for Random Forest here
            # Make prediction
            prob_categorized = nrf_cat_model.predict_proba(rf_input_categorical_df)[0, 1]
            prob_continuous = nrf_cont_model.predict_proba(rf_input_continuous_df)[0, 1]

            # Display results in the placeholder
            output_placeholder.success(
                f"""
                #### Predicted Probability:
                
                - **Model based on categorized values**: {prob_categorized:.2%}
                - **Model based on continuous values**: {prob_continuous:.2%}
                """
            )

            # Example: prediction = random_forest_model.predict(input_data)

        elif model_choice == "Logistic Regression Model":
            lr_input_continuous = {
                "Intercept": 1,  # Always included
                "ADL.dependent": int(adl_dependent),
                "AF": int(atrial_fibrillation),
                "age": age,
                "albumin": albumin,
                "calcium.total.serum": calcium,
                "CCI": cci,
                "CHF.merge": int(heart_failure),
                "CRRT.given": int(crrt),
                "CVA": int(cva),
                "dementia": int(dementia),
                "eGFR": egfr,
                "haemoglobin": haemoglobin,
                "liver.disease": int(liverdisease),
                "MI.NSTEMI": int(mi_nstemi),
                "phosphate.inorganic.serum": phosphate,
                "PVD": int(pvd),
                "raceIndian": raceIndian,
                "raceMalay": raceMalay,
                "raceOthers": raceOthers,
            }

            lr_input_continuous_df = pd.DataFrame([lr_input_continuous])

            # Insert model-specific logic for Logistic Regression here

            prob_continuous = 1 / (1 + np.exp(-np.dot(lr_input_continuous_df.iloc[0], list(continuous_log_reg_coefficients.values()))))

            # Display results in the placeholder
            output_placeholder.success(
                f"""
                #### Predicted Probability:
                
                - **Model based on categorized values**: not coded yet
                - **Model based on continuous values**: {prob_continuous:.2%}
                """
            )
            #{prob_categorized:.2%}

