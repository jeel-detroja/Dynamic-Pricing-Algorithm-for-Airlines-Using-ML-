import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Attempt to import xgboost, fail gracefully if not installed
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ---------------------------------------------------------
# 1. Feature Engineering Function
# ---------------------------------------------------------
def engineer_features(df):
    """Generates advanced aviation-specific features."""
    df = df.copy()
    
    # Pressure & Velocity Metrics
    df['seat_pressure'] = df['seats_sold'] / df['total_seats']
    df['demand_pressure'] = df['demand_index'] * df['route_popularity_score']
    
    # Booking Velocity (Avoid division by zero)
    df['booking_velocity'] = np.where(
        df['booking_trend_last_7_days'] == 0, 
        0, 
        df['booking_trend_last_3_days'] / df['booking_trend_last_7_days']
    )
    
    # Time to Departure Bucket
    conditions = [
        df['days_before_departure'] <= 7,
        (df['days_before_departure'] > 7) & (df['days_before_departure'] <= 30),
        df['days_before_departure'] > 30
    ]
    choices = ['Short', 'Medium', 'Long']
    df['time_to_departure_bucket'] = np.select(conditions, choices, default='Long')
    
    # Demand-Supply Ratio
    df['demand_supply_ratio'] = df['demand_index'] / (df['seats_remaining'] + 1)
    
    return df

# ---------------------------------------------------------
# 2. Main Training Pipeline
# ---------------------------------------------------------
def main():
    print("Loading data...")
    # Adjust path assuming execution from project root
    df = pd.read_csv('data/airline_dynamic_pricing_dataset.csv')
    
    print("Engineering features...")
    df = engineer_features(df)
    
    # Drop columns that are IDs, dates, or cause data leakage
    # dynamic_price_multiplier causes leakage because final_price = base * multiplier
    drop_cols = ['flight_id', 'departure_date', 'booking_date', 'dynamic_price_multiplier']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    # Define Target and Features
    target = 'final_ticket_price'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Identify Categorical and Numerical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ---------------------------------------------------------
    # 3. Model Selection & Evaluation
    # ---------------------------------------------------------
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBRegressor(random_state=42, n_jobs=-1)
        
    best_model_name = None
    best_model_score = -float('inf')
    best_pipeline = None
    
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"{name}: R2 = {r2:.4f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}")
        
        if r2 > best_model_score:
            best_model_score = r2
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest Model Selected: {best_model_name} (R2: {best_model_score:.4f})")
    
    # ---------------------------------------------------------
    # 4. Hyperparameter Tuning (Example on Random Forest)
    # ---------------------------------------------------------
    if best_model_name == 'Random Forest':
        print("\n--- Tuning Random Forest ---")
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        }
        
        # Reduced cross-validation folds for speed
        grid_search = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        
        y_pred_tuned = best_pipeline.predict(X_test)
        print(f"Tuned R2: {r2_score(y_test, y_pred_tuned):.4f}")

    # ---------------------------------------------------------
    # 5. Save the Model
    # ---------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    model_path = 'models/dynamic_pricing_model.pkl'
    joblib.dump(best_pipeline, model_path)
    print(f"\nModel successfully saved to {model_path}")

    # ---------------------------------------------------------
    # 6. Feature Importance Extract (for RF/GBM/XGB)
    # ---------------------------------------------------------
    if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
        importances = best_pipeline.named_steps['regressor'].feature_importances_
        # Get feature names from preprocessor
        cat_features = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(cat_features)
        
        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df, x='Importance', y='Feature', hue='Feature', legend=False, palette='viridis')
        plt.title('Top 15 Feature Importances')
        os.makedirs('outputs/plots', exist_ok=True)
        plt.savefig('outputs/plots/feature_importance.png', bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved to outputs/plots/feature_importance.png")

if __name__ == '__main__':
    main()