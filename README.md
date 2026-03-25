✈️ Airline Dynamic Pricing Machine Learning System

📌 Project Overview

This project is an end-to-end Machine Learning pipeline that simulates a realistic Airline Revenue Management system. It predicts optimal flight ticket prices dynamically based on demand, seat inventory, booking velocity, and market conditions using a highly tuned XGBoost regression model.

The system features a production-ready REST API backend and an interactive Streamlit frontend for real-time price simulations.

🏗️ System Architecture

Data Pipeline: Preprocesses synthetic airline data, handles missing values, and engineers aviation-specific features (e.g., seat_pressure, booking_velocity).

ML Model: Evaluated multiple models (Linear Regression, Random Forest, Gradient Boosting, XGBoost). Selected XGBoost for its superior performance ($R^2$: 0.9976).

Pricing Engine: A Python inference module that applies business logic (minimum price floors, maximum surge caps) on top of the ML predictions.

Backend API: Built with FastAPI to serve the pricing engine as a scalable microservice.

Frontend Dashboard: Built with Streamlit to allow users to interact with the pricing engine via a web UI.

📊 Key Business Insights (Model Interpretability)

Based on our SHAP analysis and Feature Importance extraction, the primary drivers of airline pricing are:

Competitor Pricing: The strongest anchor for our dynamic pricing.

Days Before Departure: Prices surge exponentially within the 7-day window prior to departure.

Demand & Seat Pressure: High route popularity combined with low remaining seats triggers our model's maximum surge multipliers.

Booking Velocity: Spikes in recent bookings (last 3 days vs last 7 days) alert the model to sudden demand shifts, raising prices before inventory depletes.

🚀 How to Run the Project

1. Install Dependencies

pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib fastapi uvicorn pydantic streamlit shap requests


2. Train the Model (Optional - Pre-trained model included)

python models/train_model.py


3. Start the API Backend

Keep this running in your terminal:

python api/app.py


API documentation available at: http://localhost:8000/docs

4. Launch the Interactive Dashboard

Open a new terminal window and run:

streamlit run streamlit_app.py


The UI will open in your default web browser.

📁 Project Structure

/api - FastAPI backend application

/data - Dataset directory

/models - Training scripts and serialized .pkl models

/notebooks - EDA, Model Evaluation, and SHAP interpretability scripts

/outputs/plots - Generated visualizations (Feature Importance, Predicted vs Actual, SHAP)

/pricing_engine - Core business logic and model inference functions

streamlit_app.py - Interactive web frontend