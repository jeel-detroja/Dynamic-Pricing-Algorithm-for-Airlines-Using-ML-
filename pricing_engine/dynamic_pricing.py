import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# 1. Load the Pre-Trained Model (Using Absolute Path)
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
# Use abspath to ensure Windows resolves the '..' perfectly
MODEL_PATH = os.path.abspath(os.path.join(base_dir, '..', 'models', 'dynamic_pricing_model.pkl'))

try:
    pricing_pipeline = joblib.load(MODEL_PATH)
    print(f"\n✅ Pricing model successfully loaded from:\n   {MODEL_PATH}\n")
except Exception as e:
    print(f"\n❌ ERROR: Could not load the model from:\n   {MODEL_PATH}")
    print(f"❌ Exception Details: {e}\n")
    pricing_pipeline = None

# ---------------------------------------------------------
# 2. Dynamic Pricing Function
# ---------------------------------------------------------
def predict_dynamic_price(
    days_before_departure: int,
    seats_remaining: int,
    demand_index: float,
    competitor_avg_price: float,
    route_popularity_score: int,
    base_ticket_price: float,
    total_seats: int = 150,
    **kwargs
) -> float:
    """
    Predicts the optimal dynamic ticket price using the trained ML model.
    Applies business logic guardrails to ensure realistic outputs.
    """
    if pricing_pipeline is None:
        raise RuntimeError("Pricing model is not loaded. Check your terminal for startup errors.")

    # 1. Calculate base parameters
    seats_sold = total_seats - seats_remaining
    
    # 2. Construct the observation dictionary
    observation = {
        'days_before_departure': days_before_departure,
        'seats_remaining': seats_remaining,
        'demand_index': demand_index,
        'competitor_avg_price': competitor_avg_price,
        'route_popularity_score': route_popularity_score,
        'base_ticket_price': base_ticket_price,
        'total_seats': total_seats,
        'seats_sold': seats_sold,
        
        # Required default fields for Pipeline schema compatibility
        'route_distance_km': kwargs.get('route_distance_km', 1000),
        'is_weekend': kwargs.get('is_weekend', 0),
        'holiday_indicator': kwargs.get('holiday_indicator', 0),
        'load_factor': (seats_sold / total_seats) * 100,
        'fuel_price_index': kwargs.get('fuel_price_index', 100.0),
        'booking_trend_last_3_days': kwargs.get('booking_trend_last_3_days', 5),
        'booking_trend_last_7_days': kwargs.get('booking_trend_last_7_days', 15),
        'historical_avg_price_route': kwargs.get('historical_avg_price_route', base_ticket_price),
        'price_volatility_index': kwargs.get('price_volatility_index', 5.0),
        
        'airline': kwargs.get('airline', 'Air India'),
        'route': kwargs.get('route', 'DEL-BOM'),
        'departure_time_category': kwargs.get('departure_time_category', 'Morning'),
        'day_of_week': kwargs.get('day_of_week', 'Monday'),
        'season': kwargs.get('season', 'Winter'),
        'ticket_class': kwargs.get('ticket_class', 'Economy')
    }
    
    # Convert to DataFrame
    df_obs = pd.DataFrame([observation])
    
    # 3. Apply Feature Engineering
    df_obs['seat_pressure'] = df_obs['seats_sold'] / df_obs['total_seats']
    df_obs['demand_pressure'] = df_obs['demand_index'] * df_obs['route_popularity_score']
    
    df_obs['booking_velocity'] = np.where(
        df_obs['booking_trend_last_7_days'] == 0, 
        0, 
        df_obs['booking_trend_last_3_days'] / df_obs['booking_trend_last_7_days']
    )
    
    conditions = [
        df_obs['days_before_departure'] <= 7,
        (df_obs['days_before_departure'] > 7) & (df_obs['days_before_departure'] <= 30),
        df_obs['days_before_departure'] > 30
    ]
    df_obs['time_to_departure_bucket'] = np.select(conditions, ['Short', 'Medium', 'Long'], default='Long')
    df_obs['demand_supply_ratio'] = df_obs['demand_index'] / (df_obs['seats_remaining'] + 1)
    
    # 4. Model Inference
    predicted_price = pricing_pipeline.predict(df_obs)[0]
    
    # 5. Apply Business Guardrails
    min_floor = base_ticket_price * 0.80
    max_surge = base_ticket_price * 3.50
    
    final_price = max(min_floor, min(predicted_price, max_surge))
    
    return round(final_price, 2)

if __name__ == '__main__':
    print("Testing Dynamic Pricing Engine...\n")
    price_high = predict_dynamic_price(
        days_before_departure=2, seats_remaining=5, demand_index=95.5,
        competitor_avg_price=15000, route_popularity_score=9, base_ticket_price=5000, total_seats=150
    )
    price_low = predict_dynamic_price(
        days_before_departure=45, seats_remaining=120, demand_index=30.0,
        competitor_avg_price=5200, route_popularity_score=4, base_ticket_price=5000, total_seats=150
    )
    print(f"Base Ticket Price: $5000.00")
    print("-" * 40)
    print(f"Scenario 1 Predicted Price: ${price_high}")
    print(f"Scenario 2 Predicted Price: ${price_low}")