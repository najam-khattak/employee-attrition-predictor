# 🧠 AttritionIQ — Employee Attrition Prediction Platform

An AI-powered HR Intelligence platform that predicts employee attrition using Machine Learning, built with Python and Streamlit.

---

## 🚀 Live Demo

👉 **[Open Live App](https://your-app-link.streamlit.app)**

---

## 📌 Project Overview

Employee attrition is one of the most costly challenges for organizations. This project builds an end-to-end machine learning pipeline to:

- **Predict** whether an employee is likely to leave the organization
- **Analyze** key factors driving attrition
- **Visualize** workforce insights through an interactive dashboard

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| ML Library | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Imbalanced Data | SMOTE (imbalanced-learn) |
| Model Saving | Joblib |

---

## 📊 Models Trained

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 85.7% | 0.70 ✅ Best |
| Random Forest | 83.3% | 0.65 |
| XGBoost | 78.2% | 0.64 |

> **Best Model:** Logistic Regression selected based on highest Macro F1 score

---

## 🗂️ Project Structure

```
Employee_Attrition_Prediction_Project/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── logger.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
│
├── streamlit_app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-predictor.git
cd employee-attrition-predictor
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Train models:**
```bash
python main.py
```

**Step 4 — Launch Streamlit app:**
```bash
streamlit run streamlit_app.py
```

---

## 📱 App Features

### 🏠 Dashboard
- KPI cards — Total Employees, Attrition Cases, Avg Income, Avg Tenure
- 10+ interactive charts — Department, OverTime, Age, Income, Job Role analysis

### 🔍 Predict Attrition
- Fill in 30 employee features
- Instant risk prediction with probability score
- Key influencing factors displayed

### 📊 Data Insights
- Correlation heatmap
- Feature distribution charts
- Attrition patterns by role, education, satisfaction

### 🤖 Model Performance
- Confusion Matrix
- ROC Curve with AUC score
- Feature Importance chart
- Full classification report

---

## 📁 Dataset

**IBM HR Analytics Employee Attrition Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Rows: 1,470 employees
- Features: 35 columns
- Target: Attrition (Yes/No)

---

## 👤 Author

**Najam Khattak**
- LinkedIn: [Your LinkedIn URL]
- GitHub: [Your GitHub URL]

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).