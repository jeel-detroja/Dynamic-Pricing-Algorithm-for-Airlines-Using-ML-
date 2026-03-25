import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Add project root to path so we can import pricing_engine if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_model import engineer_features

def evaluate():
    print("Loading data and model...")
    df = pd.read_csv('../data/airline_dynamic_pricing_dataset.csv')
    model = joblib.load('../models/dynamic_pricing_model.pkl')

    # Engineer features just like in training
    df_engineered = engineer_features(df)
    
    # Take a random sample of 200 flights to make the scatter plot readable
    sample_df = df_engineered.sample(n=200, random_state=42).copy()
    
    # Actual prices
    y_actual = sample_df['final_ticket_price']
    
    # Predict using the model
    # Note: We pass the whole dataframe; the pipeline's ColumnTransformer will select what it needs
    y_pred = model.predict(sample_df)
    
    # Plotting Predicted vs Actual
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.7, color='dodgerblue', edgecolor='k')
    
    # Plot the perfect prediction diagonal line
    max_val = max(y_actual.max(), y_pred.max())
    min_val = min(y_actual.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.title('Predicted vs Actual Ticket Prices')
    plt.xlabel('Actual Final Ticket Price ($)')
    plt.ylabel('Predicted Final Ticket Price ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs('../outputs/plots', exist_ok=True)
    plt.savefig('../outputs/plots/predicted_vs_actual.png', bbox_inches='tight')
    print("Evaluation complete. Check outputs/plots/predicted_vs_actual.png")

if __name__ == "__main__":
    evaluate()