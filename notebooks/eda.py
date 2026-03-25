# %% [markdown]
# # Airline Dynamic Pricing - Exploratory Data Analysis (EDA)
# This script can be executed as a standard Python file or run in Jupyter/VS Code using interactive windows.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs directory if it doesn't exist
os.makedirs('../outputs/plots', exist_ok=True)

# %%
# 1. Load the Dataset
# Assuming the script is run from the notebooks/ directory
data_path = '../data/airline_dynamic_pricing_dataset.csv'
try:
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Dataset not found at {data_path}. Please ensure the dataset is placed correctly.")
    # Creating a small mock dataframe strictly for code execution safety if file is missing during test
    df = pd.DataFrame(np.random.rand(100, 5), columns=['final_ticket_price', 'seats_remaining', 'days_before_departure', 'demand_index', 'route_popularity_score'])

# %%
# 2. Data Preprocessing for EDA
# Drop duplicates
df.drop_duplicates(inplace=True)

# %%
# 3. Feature Engineering (EDA Scope)
if 'seats_sold' in df.columns and 'total_seats' in df.columns:
    df['load_factor_calc'] = (df['seats_sold'] / df['total_seats']) * 100
    df['seat_pressure'] = df['seats_sold'] / df['total_seats']

if 'demand_index' in df.columns and 'route_popularity_score' in df.columns:
    df['demand_pressure'] = df['demand_index'] * df['route_popularity_score']

# %%
# 4. Visualizations
sns.set_theme(style="whitegrid")

# Plot 1: Ticket Price vs Seats Remaining
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='seats_remaining', y='final_ticket_price', alpha=0.5, color='b')
plt.title('Ticket Price vs Seats Remaining')
plt.xlabel('Seats Remaining')
plt.ylabel('Final Ticket Price')
plt.gca().invert_xaxis() # Fewer seats -> Right side
plt.savefig('../outputs/plots/price_vs_seats.png')
plt.close()

# Plot 2: Ticket Price vs Days Before Departure
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='days_before_departure', y='final_ticket_price', color='r')
plt.title('Ticket Price vs Days Before Departure')
plt.xlabel('Days Before Departure')
plt.ylabel('Average Ticket Price')
plt.gca().invert_xaxis() # Closer to departure -> Right side
plt.savefig('../outputs/plots/price_vs_days.png')
plt.close()

# Plot 3: Ticket Price vs Demand Index
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='demand_index', y='final_ticket_price', hue='season', palette='viridis', alpha=0.6)
plt.title('Ticket Price vs Demand Index by Season')
plt.xlabel('Demand Index')
plt.ylabel('Final Ticket Price')
plt.savefig('../outputs/plots/price_vs_demand.png')
plt.close()

# Plot 4: Correlation Heatmap
plt.figure(figsize=(14, 10))
# Select only numerical columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig('../outputs/plots/correlation_heatmap.png')
plt.close()

print("EDA complete. Visualizations saved in outputs/plots/")