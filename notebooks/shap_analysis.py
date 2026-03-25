import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_model import engineer_features

def run_shap_analysis():
    print("Loading data and model for SHAP analysis...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'airline_dynamic_pricing_dataset.csv')
    model_path = os.path.join(base_dir, '..', 'models', 'dynamic_pricing_model.pkl')
    
    df = pd.read_csv(data_path)
    pipeline = joblib.load(model_path)

    # Engineer features and sample data (SHAP is computationally heavy, so we use a sample)
    df_engineered = engineer_features(df)
    
    drop_cols = ['final_ticket_price', 'flight_id', 'departure_date', 'booking_date', 'dynamic_price_multiplier']
    X = df_engineered.drop(columns=[col for col in drop_cols if col in df_engineered.columns])
    
    # Take a representative sample of 500 rows
    X_sample = X.sample(n=500, random_state=42)

    print("Extracting pipeline components...")
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['regressor']

    # Transform the data using the preprocessor
    X_transformed = preprocessor.transform(X_sample)
    
    # Get feature names from the preprocessor
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_features)

    print("Calculating SHAP values (this may take a moment)...")
    # Use TreeExplainer for XGBoost/RandomForest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    
    plot_dir = os.path.join(base_dir, '..', 'outputs', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, 'shap_summary_plot.png')
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP analysis complete! Saved to {save_path}")

if __name__ == "__main__":
    run_shap_analysis()