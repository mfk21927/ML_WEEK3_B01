# ML_WEEK3_B01
# 🚀 ML-Internship

> **Name:** Muhammad Fahad  
> **Email:** mfk21927@gmail.com  
> **Start Date:** 20-12-2025

---

![Internship](https://img.shields.io/badge/Status-Active-blue?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-B01-orange?style=for-the-badge)

## 📂 Table of Contents
* [📌 Project Overview](#-project-overview)
* [📈 Weekly Progress](#-weekly-progress)
* [✅ Completed Tasks](#-completed-tasks)
* [💻 Tech Stack](#-tech-stack)
* [📫 Contact](#-contact)

---

## 📌 Project Overview
This project documents my journey through the Machine Learning Internship. It covers everything from foundational Git workflows to building, optimizing, and deploying regression models.

## 📈 Weekly Progress

| Week | Title | Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Week 3** | Introduction to ML | Regression & Persistence | ✅ Completed |

---

## ✅ Completed Tasks (Week 3)

<details>
<summary><b>Task 3.1: Simple Linear Regression from Scratch</b></summary>

- Built a custom `LinearRegression` class using **Gradient Descent**.
- Calculated $R^{2}$ score manually to verify mathematical accuracy.
- **Key Equation:** $\hat{y} = wx + b$
- **Convergence:** Visualized how the Mean Squared Error (MSE) decreased over iterations.



</details>

<details>
<summary><b>Task 3.2: Multiple Linear Regression (California Housing)</b></summary>

- Used `sklearn` to predict housing prices.
- **Proper Step:** Applied `StandardScaler` to ensure all features (income, population, etc.) were on the same scale for optimal model performance.
- **Evaluation:**
  - **MAE:** 0.5332 | **RMSE:** 0.7456
  - **$R^{2}$ Score:** ~0.5758
- **Visualization:** Created Actual vs. Predicted scatter plots and Residual Plots to check for error patterns.



</details>

<details>
<summary><b>Task 3.3: Polynomial Regression & Overfitting</b></summary>

- Tested model complexity using degrees 1, 3, and 10.
- **Underfitting (Degree 1):** Too simple; fails to capture the data trend.
- **Overfitting (Degree 10):** Extremely low training error but high test error; captures noise instead of signal.
- **Sweet Spot (Degree 3):** Balanced model providing the lowest Test RMSE.



</details>

<details>
<summary><b>Task 3.4: Model Persistence (Saving & Loading)</b></summary>

Comparison of model serialization formats for the California Housing model:

| Format | File Size | Loading Time | Accuracy Check |
| :--- | :--- | :--- | :--- |
| **Pickle (.pkl)** | ~800 B | 0.0001s | Exact Match |
| **Joblib (.joblib)**| ~900 B | 0.0000s | Exact Match |
| **JSON (Manual)** | ~250 B | 0.0005s | Exact Match |

- **Observation:** `joblib` is highly efficient for models with large arrays. `JSON` is best for language-independent weight storage but requires manual reconstruction of the prediction formula.



</details>

## 💻 Tech Stack
* **Languages:** Python, Markdown
* **Libraries:** NumPy, Scikit-Learn, Matplotlib, Pandas
* **Tools:** Git, VS Code, Joblib, Pickle

## 📫 Contact
**Muhammad Fahad**
* GitHub: [mfk21927](https://github.com/mfk21927)
* Repository: [ML_Internship_B01](https://github.com/mfk21927/ML_Internship_B01)

---
## 📜 License
This project is licensed under the MIT License.