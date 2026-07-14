# 🏥 Medical Insurance Cost Prediction

Predicting annual medical insurance costs using Machine Learning Regression models.

---

## 📌 Project Overview

This project aims to predict **Annual Medical Cost** using patient demographics, medical history, lifestyle factors, and insurance-related information.

The project follows a complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and model evaluation.

The primary objective is to understand how different regression algorithms perform on this dataset while gaining hands-on experience with the complete machine learning pipeline.

---

## 📊 Dataset Information

- **Dataset Size:** 100,000 Records
- **Target Variable:** `annual_medical_cost`
- **Dataset Type:** Regression
- **Source:** Kaggle

The dataset contains information such as:

- Age
- BMI
- Income
- Smoking Status
- Alcohol Consumption
- Blood Pressure
- Medical History
- Hospital Visits
- Chronic Diseases
- Insurance Information
- Lifestyle Factors
- And several engineered categorical features.

---

# 🛠 Data Preprocessing

The following preprocessing steps were performed:

- ✔ Removed unnecessary columns
- ✔ Handled missing values
- ✔ Removed duplicate records
- ✔ Converted categorical features using:
  - One-Hot Encoding
  - Ordinal Encoding
- ✔ Feature Scaling using **StandardScaler**
- ✔ Train-Test Split (80:20)

---

# 📈 Exploratory Data Analysis (EDA)

Several visualization techniques were used to better understand the dataset:

- 📊 Histograms
- 📦 Boxplots
- 🔥 Correlation Heatmap
- 📈 Scatter Plots

EDA helped identify:

- Feature distributions
- Outliers
- Correlations
- Potential data leakage
- Feature relationships

Several highly correlated leakage features were removed before model training.

---

# 🤖 Models Implemented

The following regression algorithms have been implemented:

- ✅ Linear Regression
- ✅ Ridge Regression
- ✅ Lasso Regression

Each model was evaluated using both the training and testing datasets.

---

# 📊 Model Performance

| Model | Train R² | Test R² |
|--------|---------:|--------:|
| Linear Regression | 0.1726 | 0.1771 |
| Ridge Regression | 0.1726 | 0.1771 |
| Lasso Regression | 0.1726 | 0.1772 |

Evaluation Metrics Used:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross Validation

---

# 📌 Key Findings

- Linear Regression provided relatively low predictive performance.
- Ridge and Lasso Regression produced nearly identical results.
- Regularization did not significantly improve the model.
- The similar training and testing scores indicate that the models were **underfitting** rather than overfitting.
- The dataset likely contains **non-linear relationships** that cannot be effectively modeled using linear algorithms.

---

# 🚀 Future Improvements

This project is currently under active development.

As I continue learning Machine Learning, I will implement additional regression algorithms on the same dataset, including:

- Random Forest Regressor
- Decision Tree Regressor
- Support Vector Regression (SVR)
- XGBoost
- Gradient Boosting
- AdaBoost
- Extra Trees Regressor

Future updates will also include:

- Hyperparameter Tuning
- Advanced Feature Engineering
- Model Comparison
- Performance Optimization

---

# 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

# 📁 Project Structure

```
Medical_Insurance_Cost_Prediction/
│
├── Medical_Insurance_Cost_Prediction.ipynb
├── README.md
├── requirements.txt
└── dataset/
```

---

# 📚 What I Learned

This project helped me gain practical experience in:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Feature Scaling
- Linear Regression
- Ridge Regression
- Lasso Regression
- Cross Validation
- Model Evaluation
- Building an End-to-End Machine Learning Pipeline

---

## ⭐ If you found this project helpful, consider giving it a star!