# ============================================================
# Employee Attrition Prediction Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import base64
import plotly.graph_objects as go

from pathlib import Path

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Load Saved Files
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "employee_attrition_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
numerical_columns = joblib.load(BASE_DIR / "numerical_columns.pkl")

# ------------------------------------------------------------
# Custom Styling
# ------------------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

h1{
    color:#0B5394;
}

.card{

    background:#F8F9FA;

    padding:18px;

    border-radius:12px;

    border:1px solid #E6E6E6;

    box-shadow:0px 2px 8px rgba(0,0,0,0.08);

}

.prediction-box{

    padding:22px;

    border-radius:14px;

    background:#EEF6FF;

    border-left:6px solid #0B5394;

}

.small-text{

    color:#777777;

    font-size:15px;

}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:

    st.title("Employee Attrition")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
        Predict employee attrition using Machine Learning.

        This dashboard analyzes employee information and estimates the likelihood of an employee leaving the organization.
        """
    )

    st.markdown("---")

    st.subheader("Workflow")

    st.write("""
    ✔ Data Cleaning

    ✔ Feature Engineering

    ✔ Model Training

    ✔ Hyperparameter Tuning

    ✔ Prediction
    """)

    st.markdown("---")

    st.subheader("Models Compared")

    st.write("""
    • Logistic Regression

    • Random Forest

    • Gradient Boosting
    """)

    st.markdown("---")

    st.info(
        "This application is intended for educational and demonstration purposes."
    )

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("Employee Attrition Prediction Dashboard")

st.markdown(
"""
Analyze employee-related attributes and predict the likelihood of attrition using a trained Machine Learning model.
"""
)

st.markdown("---")

# ------------------------------------------------------------
# Dashboard Statistics
# ------------------------------------------------------------

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "Dataset",
        "1470 Records"
    )

with metric2:
    st.metric(
        "Features",
        "31"
    )

with metric3:
    st.metric(
        "Models",
        "3"
    )

with metric4:
    st.metric(
        "Prediction",
        "Binary"
    )

st.markdown("---")

# ------------------------------------------------------------
# Project Overview
# ------------------------------------------------------------

st.info(
"""
This dashboard predicts whether an employee is likely to leave the company using a Machine Learning model trained on the IBM HR Analytics dataset.

Fill in the employee information below and click **Predict Attrition** to view the prediction, confidence score, and HR recommendations.
"""
)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# Employee Information
# ============================================================

st.header("Employee Information")

st.write(
    "Provide the employee details below to estimate the likelihood of attrition."
)

# ============================================================
# Personal Information
# ============================================================

with st.expander("👤 Personal Information", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider("Age",18,60,30)

        gender = st.selectbox(
            "Gender",
            ["Male","Female"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single","Married","Divorced"]
        )

    with col2:

        education = st.selectbox(
            "Education",
            [1,2,3,4,5]
        )

        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ]
        )

    with col3:

        distance_from_home = st.slider(
            "Distance From Home",
            1,
            30,
            5
        )

# ============================================================
# Professional Information
# ============================================================

with st.expander("💼 Professional Information", expanded=True):

    col1,col2,col3 = st.columns(3)

    with col1:

        department = st.selectbox(
            "Department",
            [
                "Research & Development",
                "Sales",
                "Human Resources"
            ]
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "Healthcare Representative",
                "Human Resources",
                "Laboratory Technician",
                "Manager",
                "Manufacturing Director",
                "Research Director",
                "Research Scientist",
                "Sales Executive",
                "Sales Representative"
            ]
        )

        job_level = st.selectbox(
            "Job Level",
            [1,2,3,4,5]
        )

    with col2:

        business_travel = st.selectbox(
            "Business Travel",
            [
                "Non-Travel",
                "Travel_Rarely",
                "Travel_Frequently"
            ]
        )

        overtime = st.selectbox(
            "OverTime",
            [
                "No",
                "Yes"
            ]
        )

        stock_option_level = st.selectbox(
            "Stock Option Level",
            [0,1,2,3]
        )

    with col3:

        job_involvement = st.select_slider(
            "Job Involvement",
            options=[1,2,3,4],
            value=3
        )

        work_life_balance = st.select_slider(
            "Work Life Balance",
            options=[1,2,3,4],
            value=3
        )

# ============================================================
# Compensation & Experience
# ============================================================

with st.expander("💰 Compensation & Experience", expanded=True):

    col1,col2,col3 = st.columns(3)

    with col1:

        monthly_income = st.number_input(
            "Monthly Income",
            1000,
            25000,
            6000,
            step=100
        )

        daily_rate = st.number_input(
            "Daily Rate",
            100,
            1500,
            800
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            30,
            100,
            60
        )

    with col2:

        monthly_rate = st.number_input(
            "Monthly Rate",
            2000,
            30000,
            15000
        )

        percent_salary_hike = st.slider(
            "Percent Salary Hike",
            10,
            25,
            15
        )

        num_companies_worked = st.slider(
            "Number of Companies Worked",
            0,
            10,
            2
        )

    with col3:

        total_working_years = st.slider(
            "Total Working Years",
            0,
            40,
            10
        )

        years_at_company = st.slider(
            "Years At Company",
            0,
            40,
            5
        )

        years_in_current_role = st.slider(
            "Years In Current Role",
            0,
            20,
            3
        )

        years_since_last_promotion = st.slider(
            "Years Since Last Promotion",
            0,
            15,
            1
        )

        years_with_curr_manager = st.slider(
            "Years With Current Manager",
            0,
            20,
            3
        )

# ============================================================
# Performance & Satisfaction
# ============================================================

with st.expander("📊 Performance & Satisfaction", expanded=True):

    col1,col2,col3 = st.columns(3)

    with col1:

        environment_satisfaction = st.select_slider(
            "Environment Satisfaction",
            options=[1,2,3,4],
            value=3
        )

        job_satisfaction = st.select_slider(
            "Job Satisfaction",
            options=[1,2,3,4],
            value=3
        )

    with col2:

        relationship_satisfaction = st.select_slider(
            "Relationship Satisfaction",
            options=[1,2,3,4],
            value=3
        )

        performance_rating = st.selectbox(
            "Performance Rating",
            [3,4]
        )

    with col3:

        training_times_last_year = st.slider(
            "Training Times Last Year",
            0,
            10,
            2
        )

# ============================================================
# Prediction Button
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_btn = st.button(
    "Predict Attrition",
    type="primary",
    use_container_width=True
)

# ============================================================
# Prediction
# ============================================================

if predict_btn:

    employee_data = {

        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EmployeeCount": 1,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,

        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime
    }

    input_df = pd.DataFrame([employee_data])

    input_df = pd.get_dummies(input_df)

    for column in feature_columns:

        if column not in input_df.columns:

            input_df[column] = 0

    input_df = input_df[feature_columns]

    input_df[numerical_columns] = scaler.transform(
        input_df[numerical_columns]
    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    probability = probability * 100

    st.markdown("---")

    st.header("Prediction Result")

    if prediction == 1:

        st.error("Employee is likely to leave the company.")

    else:

        st.success("Employee is likely to stay with the company.")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Attrition Probability",
            f"{probability:.2f}%"
        )

    with col2:

        if probability >= 70:

            risk = "High"

        elif probability >= 40:

            risk = "Medium"

        else:

            risk = "Low"

        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        confidence = max(probability,100-probability)

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    st.markdown("---")

    st.subheader("HR Recommendation")

    if risk == "High":

        st.warning(
            """
            • Conduct a retention discussion.

            • Review workload and overtime.

            • Discuss career growth opportunities.

            • Evaluate compensation and benefits.

            • Schedule regular manager feedback sessions.
            """
        )

    elif risk == "Medium":

        st.info(
            """
            • Monitor employee engagement.

            • Provide learning opportunities.

            • Review work-life balance.

            • Encourage manager interaction.
            """
        )

    else:

        st.success(
            """
            • Employee currently shows a low risk of attrition.

            • Continue performance recognition.

            • Maintain employee engagement.

            • Encourage continuous development.
            """
        )

    st.markdown("---")

    st.subheader("Employee Summary")

    summary = pd.DataFrame({

        "Feature":[
            "Age",
            "Department",
            "Job Role",
            "Monthly Income",
            "Years At Company",
            "Work Life Balance",
            "Job Satisfaction",
            "OverTime"
        ],

        "Value":[
            age,
            department,
            job_role,
            monthly_income,
            years_at_company,
            work_life_balance,
            job_satisfaction,
            overtime
        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # ==========================================================
    # Prediction Gauge
    # ==========================================================
    
    st.subheader("Prediction Confidence")
    
    fig = go.Figure(go.Indicator(
    
        mode="gauge+number",
    
        value=probability,
    
        title={"text":"Attrition Probability (%)"},
    
        gauge={
    
            "axis":{"range":[0,100]},
    
            "bar":{"color":"darkblue"},
    
            "steps":[
    
                {"range":[0,40],"color":"#90EE90"},
    
                {"range":[40,70],"color":"#FFD966"},
    
                {"range":[70,100],"color":"#FF6B6B"}
    
            ]
    
        }
    
    ))
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    st.subheader("Risk Level")
    
    st.progress(probability/100)
    
    if probability >= 70:
    
        st.error("High Attrition Risk")
    
    elif probability >= 40:
    
        st.warning("Moderate Attrition Risk")
    
    else:
    
        st.success("Low Attrition Risk")
    
    st.subheader("Employee Profile")
    
    col1,col2,col3 = st.columns(3)
    
    with col1:
    
        st.metric("Age",age)
    
        st.metric("Department",department)
    
        st.metric("Job Role",job_role)
    
    with col2:
    
        st.metric("Income",f"₹ {monthly_income}")
    
        st.metric("Years at Company",years_at_company)
    
        st.metric("Business Travel",business_travel)
    
    with col3:
    
        st.metric("Job Satisfaction",job_satisfaction)
    
        st.metric("Work Life Balance",work_life_balance)
    
        st.metric("OverTime",overtime)
    
    st.subheader("Most Important Features")
    
    if hasattr(model,"feature_importances_"):
    
        importance = pd.DataFrame({
    
            "Feature":feature_columns,
    
            "Importance":model.feature_importances_
    
        })
    
        importance = importance.sort_values(
    
            by="Importance",
    
            ascending=False
    
        ).head(10)
    
        st.bar_chart(
    
            importance.set_index("Feature")
    
        )
    
    
    st.subheader("Download Prediction Report")
    
    report = pd.DataFrame({
    
        "Field":[
    
            "Prediction",
    
            "Probability",
    
            "Risk Level",
    
            "Department",
    
            "Job Role",
    
            "Monthly Income"
    
        ],
    
        "Value":[
    
            "Leave" if prediction else "Stay",
    
            f"{probability:.2f}%",
    
            risk,
    
            department,
    
            job_role,
    
            monthly_income
    
        ]
    
    })
    
    csv = report.to_csv(index=False).encode("utf-8")
    
    st.download_button(
    
        label="Download CSV Report",
    
        data=csv,
    
        file_name="employee_attrition_prediction.csv",
    
        mime="text/csv"
    
    )
    
    st.subheader("Input Summary")
    
    summary = pd.DataFrame({
    
        "Feature":[
    
            "Age",
    
            "Department",
    
            "Job Role",
    
            "Income",
    
            "Years At Company",
    
            "OverTime",
    
            "Job Satisfaction",
    
            "Work Life Balance"
    
        ],
    
        "Value":[
    
            age,
    
            department,
    
            job_role,
    
            monthly_income,
    
            years_at_company,
    
            overtime,
    
            job_satisfaction,
    
            work_life_balance
    
        ]
    
    })
    
    st.dataframe(
    
        summary,
    
        use_container_width=True,
    
        hide_index=True
    )
    
    st.markdown("---")
    
    st.caption(
        "Employee Attrition Prediction Dashboard • Built with Streamlit & Scikit-learn"
    )
    
