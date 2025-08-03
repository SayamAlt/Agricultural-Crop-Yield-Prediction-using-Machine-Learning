# 🌾 Agricultural Crop Yield Prediction using Machine Learning in R

## 🚀 Project Overview

Successfully developed a **crop yield regression system** using advanced machine learning algorithms in **R**. This project aims to **predict crop yields (in tons per hectare)** using agro-environmental, soil, and weather data. The solution includes a **fully automated pipeline** for data preprocessing, model training, and performance evaluation—tailored for practical decision-making in agriculture.

---

## 🔑 Key Features

- **Exploratory Data Analysis (EDA):**  
  Comprehensive visualizations using `ggplot2`, `corrplot`, and `DataExplorer` for pattern recognition and outlier detection.

- **Feature Engineering:**
  - One-hot encoding for categorical variables
  - Handling missing values and constant columns
  - Ensured numerical format compatibility for ML models

- **Feature Scaling:**  
  Min-max normalization with protection against division-by-zero for constant features.

- **Model Building:**  
  Implemented and evaluated **20+ regression models**, including:
  - Linear Regression, Ridge, Lasso  
  - Decision Tree, Random Forest, Bagging  
  - Gradient Boosting, XGBoost, LightGBM, CatBoost  
  - K-Nearest Neighbors (KNN), Support Vector Machines (SVM)  
  - Bayesian Regression, Quantile Regression  
  - Neural Networks (`nnet`, `neuralnet`)

- **Hyperparameter Tuning:**  
  Optimized models using `caret`, `xgboost::xgb.cv`, `lgb.cv`, and `randomForest::tuneRF`.

- **Performance Metrics:**  
  RMSE, MAE, MAPE, R², Adjusted R², and RMSLE.

- **Model Comparison:**  
  Side-by-side metric comparison to select the best-performing algorithm.

- **Interpretability:**
  - Variable importance via `varImp`, `xgb.importance`, `plotnet`, `olden`, `garson`
  - Uncertainty estimation using **Quantile Regression**

---

## 📊 Results

- **Top-performing models included:**
  - ⭐ **Linear Regression** (R² = 0.9132)
  - ⭐ **Tuned GAM / Generalized Additive Model** (R² = 0.9128)
  - ⭐ **Gaussian GLM** (R² = 0.9128)

- **Other strong performers:**  
  Tuned **CatBoost**, **XGBoost**, **LightGBM**, and **Neural Networks** also achieved **R² > 0.91** after tuning.

- **Underperformers:**  
  Simpler models like **Decision Trees**, **Random Forests**, and **KNN** lagged behind.  
  **Bayesian Regression**, **Lasso**, **Ridge**, and **Elastic Net** showed **poor generalization**.

---

## ⚙️ Tech Stack

- **Language:** R  
- **Libraries:**  
  `caret`, `xgboost`, `lightgbm`, `catboost`, `randomForest`, `rpart`, `neuralnet`, `nnet`, `Metrics`, `quantreg`, `ggplot2`, `dplyr`, `glmnet`, `e1071`, `DataExplorer`, `mlbench`, `MASS`, `corrplot`, `smotefamily`

---

## 🏆 Use Cases

- Government and NGOs for **agricultural planning** and **subsidy allocation**
- Agribusiness firms for **forecasting crop production** and optimizing **supply chain logistics**
- Farm-level **decision support systems** for precision agriculture

---

## 📈 Future Enhancements

- Deploy via **Shiny dashboard** for interactive yield prediction tools  
- Incorporate **remote sensing** and **real-time weather data**  
- Build a **REST API** for integration into farm management systems