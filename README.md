# 📊 Medical Insurance Charges EDA

This project performs an in-depth Exploratory Data Analysis (EDA) on a health insurance dataset to understand what factors influence medical charges. The goal is to identify patterns, correlations, and insights that can later help in building predictive models.

---

## 📁 Dataset

The dataset contains **demographic and health-related attributes** of insurance policyholders, including:

- `age`: Age of the individual
- `sex`: Gender (`male`, `female`)
- `bmi`: Body Mass Index
- `children`: Number of children covered by the insurance
- `smoker`: Smoking status (`yes`, `no`)
- `region`: Residential region in the U.S. (`southeast`, `southwest`, etc.)
- `charges`: Medical insurance cost billed

---

## 🔍 Objectives

- Identify the most influential variables affecting `charges`
- Study distribution and skewness of continuous variables
- Compare average charges by `age group`, `smoker status`, `region`, etc.
- Check for correlation and multicollinearity
- Prepare the data for machine learning pipeline

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (for preprocessing)

---

## 📈 Key Insights

- **Smokers** are charged significantly more (~3–4x) than non-smokers.
- **BMI** is positively correlated with charges, especially beyond the obesity threshold (BMI > 30).
- **Age** has a non-linear impact; charges tend to increase sharply after age 40.
- **Region** has some impact, with the southeast generally showing higher charges.

---

## 📂 Project Structure

📁 insurance-eda-project
│
├── eda.ipynb ← Main EDA notebook
├── insurance.csv ← Dataset
└── README.md ← Project overview (this file
