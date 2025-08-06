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

📊 Model Performance
The following table summarizes model performance before and after hyperparameter tuning.

🔹 Baseline Results (Default Parameters)
| Model             | R² Score | Adj. R² | RMSE    | MAPE   |
| ----------------- | -------- | ------- | ------- | ------ |
| Linear            | 0.5124   | 0.5070  | 8651.08 | 0.2939 |
| Ridge             | 0.5157   | 0.5104  | 8621.17 | 0.2942 |
| Lasso             | 0.5164   | 0.5111  | 8615.02 | 0.4507 |
| Random Forest     | 0.9370   | 0.9363  | 3109.88 | 0.1089 |
| Gradient Boosting | 0.8608   | 0.8593  | 4622.40 | 0.1610 |
| SVR               | 0.8145   | 0.8125  | 5335.42 | 0.1462 |
| LightGBM          | 0.8864   | 0.8852  | 4174.83 | 0.1636 |
| XGBoost           | 0.8559   | 0.8543  | 4703.24 | 0.1641 |


🔹 After Hyperparameter Tuning
| Model             | Best Params                                                         | R² Score | Adj. R² | RMSE     | MAPE   |
| ----------------- | ------------------------------------------------------------------- | -------- | ------- | -------- | ------ |
| Linear            | `{}`                                                                | 0.7398   | 0.7370  | 6319.27  | 0.4617 |
| Ridge             | `{'regressor__alpha': 1.0}`                                         | 0.7397   | 0.7369  | 6320.10  | 0.4626 |
| Lasso             | `{'regressor__alpha': 0.1}`                                         | 0.7398   | 0.7370  | 6319.27  | 0.4617 |
| Random Forest     | `{'regressor__max_depth': 10, 'regressor__n_estimators': 300}`      | 0.9360   | 0.9353  | 3135.22  | 0.2051 |
| Gradient Boosting | `{'regressor__learning_rate': 0.1, 'regressor__n_estimators': 300}` | 0.9361   | 0.9354  | 3131.18  | 0.2260 |
| SVR               | `{'regressor__C': 10, 'regressor__gamma': 'scale'}`                 | -0.0334  | -0.0447 | 12594.02 | 0.9496 |
| LightGBM          | `{'regressor__learning_rate': 0.1, 'regressor__n_estimators': 300}` | 0.9396   | 0.9389  | 3045.46  | 0.2818 |
| XGBoost           | `{'regressor__learning_rate': 0.1, 'regressor__n_estimators': 300}` | 0.9222   | 0.9214  | 3455.38  | 0.2443 |

✅ Insights
LightGBM and Random Forest achieved the highest R² scores after tuning.

Linear models (even after tuning) couldn't capture complex non-linearities in medical charges.

SVR performed very poorly after tuning — indicating it is unsuitable for this problem without further preprocessing or feature engineering.

Tree-based models are consistently the best performers for this regression task.



## 📂 Project Structure

📁 insurance-eda-project
│
├── eda.ipynb ← Main EDA notebook
├── insurance.csv ← Dataset
└── README.md ← Project overview (this file
