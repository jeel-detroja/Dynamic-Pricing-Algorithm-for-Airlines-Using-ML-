# 🤖 Dynamic Pricing Algorithm for Airlines

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)

An end-to-end Machine Learning system designed to predict optimal airline ticket prices. This project covers the entire ML lifecycle—from generating synthetic datasets and engineering features to training a predictive model and deploying it via a REST API with an interactive dashboard.

---

## 📋 Table of Contents
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Workflow](#-workflow)
- [Installation & Setup](#️-how-to-run-step-by-step)
- [Project Output](#-output)
- [License](#-license)

---

## 🚀 Features
- **Predictive Modeling:** Accurately predict ticket prices based on demand, route, timing, and other key factors.
- **Full ML Lifecycle:** Includes data preprocessing, feature engineering, and model evaluation.
- **REST API Integration:** A fast, robust backend built to serve model predictions in real-time.
- **Interactive Dashboard:** A user-friendly web interface to simulate real-world airline pricing and visualize trends.
- **Explainable AI:** Model interpretability and insights into which factors drive price changes.

---

## 🛠️ Tech Stack
- **Machine Learning:** Scikit-learn, XGBoost, Joblib, SHAP
- **Data Processing:** Pandas, NumPy
- **Backend API:** FastAPI, Uvicorn, Pydantic, Requests
- **Frontend / Dashboard:** Streamlit
- **Data Visualization:** Matplotlib, Seaborn
- **Environment:** Python, Jupyter Notebook

---

## 📊 Workflow
1. **Data Collection:** Generating/loading the dataset.
2. **Data Cleaning:** Handling missing values and formatting data types.
3. **Feature Engineering:** Creating meaningful variables (e.g., days to departure, seasonality).
4. **Model Training:** Training and tuning the ML algorithms (e.g., XGBoost).
5. **Deployment:** Serving predictions via FastAPI.
6. **Visualization:** Consuming the API through a Streamlit frontend.

---

## ⚙️ How to Run (Step-by-Step)

### 1️⃣ Clone the Repository
```bash
git clone [https://github.com/jeel-detroja/dynamic-pricing-airlines.git](https://github.com/jeel-detroja/dynamic-pricing-airlines.git)
cd dynamic-pricing-airlines
```

### 2️⃣ Create and Activate a Virtual Environment (Recommended)
- **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

### 3️⃣ Install Dependencies
Install all required libraries for the ML model, API, and dashboard:
```bash
pip install -r requirements.txt
```
*(Note: If you don't have a `requirements.txt` yet, you can run: `pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib fastapi uvicorn pydantic streamlit shap requests`)*

### 4️⃣ Train the Model (Optional)
If you want to retrain the model from scratch (a pre-trained model is already included):
```bash
python models/train_model.py
```

### 5️⃣ Start the API Backend
Keep this terminal window open and running. This starts the FastAPI server that serves your predictions:
```bash
python api/app.py
```
*API documentation (Swagger UI) will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)*

### 6️⃣ Launch the Interactive Dashboard
Open a **new** terminal window (make sure your virtual environment is activated there too), and run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
*The dashboard will automatically open in your default web browser.*

---

## 📈 Output
- **Predicted Ticket Prices:** Real-time price estimations based on user inputs.
- **Graphs and Insights:** Visual breakdowns of pricing trends, feature importance, and historical data patterns.

---

## 📄 License
This project is licensed under the MIT License.

---
*Built by [jeel-detroja](https://github.com/jeel-detroja)*
