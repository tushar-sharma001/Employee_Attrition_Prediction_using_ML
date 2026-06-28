# 📊 Employee Attrition Prediction using Machine Learning

Predict employee attrition using Machine Learning to help HR teams identify employees at risk of leaving and make data-driven retention decisions.

---

## 📌 Overview

Employee attrition is a major challenge for organizations, leading to increased hiring costs, reduced productivity, and knowledge loss. This project builds an end-to-end Machine Learning pipeline that predicts whether an employee is likely to leave the organization based on various HR-related factors.

The project covers the complete Data Science workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, hyperparameter tuning, explainability, and business recommendations.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Multiple Machine Learning Models
- Hyperparameter Tuning using GridSearchCV
- Cross Validation
- Model Evaluation
- Feature Importance Analysis
- Explainable AI (Permutation Importance)
- Model Serialization
- Employee Attrition Prediction
- HR Business Insights & Recommendations

---

## 📂 Dataset

**Dataset:** IBM HR Analytics Employee Attrition Dataset

- 1,470 Employee Records
- 35 Features
- Binary Classification Problem

Target Variable:

```
Attrition
0 → Employee Stayed
1 → Employee Left
```

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📊 Exploratory Data Analysis

The following analyses were performed:

- Employee Attrition Distribution
- Department-wise Attrition
- Job Role Analysis
- Monthly Income vs Attrition
- Work-Life Balance Analysis
- Years at Company Analysis
- Job Satisfaction
- Environment Satisfaction
- Business Travel
- Overtime Analysis
- Correlation Heatmap
- Feature Correlation with Attrition

---

## 🤖 Machine Learning Models

The following classification models were trained and compared:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Hyperparameter tuning was performed using GridSearchCV.

---

## 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Cross Validation

---

## 📉 Visualizations

- Attrition Distribution
- Department-wise Attrition
- Job Role-wise Attrition
- Monthly Income Box Plot
- Work-Life Balance Analysis
- Years at Company Analysis
- Correlation Heatmap
- ROC Curve
- Confusion Matrix
- Feature Importance

---

## 🔍 Explainability

The project includes model interpretation using:

- Feature Importance
- Permutation Importance

These techniques help identify the factors contributing most to employee attrition.

---

## 💡 Key Business Insights

- Employees working overtime are more likely to leave.
- Lower job satisfaction is associated with higher attrition.
- Employees in the early years of employment show higher exit rates.
- Certain job roles experience significantly higher attrition.
- Salary influences attrition but is not the only deciding factor.

---

## 🎯 Business Recommendations

- Identify high-risk employees early using predictive analytics.
- Improve work-life balance for employees working overtime.
- Focus retention strategies on departments with high attrition.
- Conduct periodic employee satisfaction surveys.
- Strengthen onboarding and mentorship programs for new employees.

---

## 📁 Project Structure

```
Employee-Attrition-Prediction/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebook.ipynb
├── app.py
├── employee_attrition_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
└── images/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Employee-Attrition-Prediction.git
```

Move into the project directory

```bash
cd Employee-Attrition-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## 📌 Future Improvements

- Deploy using Streamlit
- SHAP Explainability
- XGBoost and LightGBM Models
- Interactive Dashboard
- Cloud Deployment
- Real-time HR Analytics Dashboard

---

## 📜 License

This project is intended for educational and learning purposes.

---

## 👨‍💻 Author

**Tushar Sharma**

BCA Student | AI & Machine Learning Enthusiast
