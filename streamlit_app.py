import streamlit as st
import requests

# Configure page
st.set_page_config(page_title="Airline Dynamic Pricing", page_icon="✈️", layout="wide")

st.title("✈️ Airline Dynamic Pricing Engine")
st.markdown("Adjust the flight parameters below to see how the ML model adjusts the ticket price dynamically.")

# Create columns for UI layout
col1, col2 = st.columns(2)

with col1:
    st.header("Flight Details")
    base_ticket_price = st.number_input("Base Ticket Price ($)", min_value=100.0, max_value=20000.0, value=5000.0, step=100.0)
    days_before_departure = st.slider("Days Before Departure", min_value=0, max_value=180, value=15)
    total_seats = st.number_input("Total Seats on Aircraft", min_value=50, max_value=500, value=150)
    seats_remaining = st.slider("Seats Remaining", min_value=0, max_value=total_seats, value=45)

with col2:
    st.header("Market Conditions")
    demand_index = st.slider("Demand Index (0-100)", min_value=0.0, max_value=100.0, value=75.0)
    route_popularity_score = st.slider("Route Popularity (1-10)", min_value=1, max_value=10, value=7)
    competitor_avg_price = st.number_input("Competitor Average Price ($)", min_value=100.0, max_value=20000.0, value=5200.0, step=100.0)

# Prediction Button
if st.button("Predict Dynamic Price", type="primary", use_container_width=True):
    # Construct the payload to send to your FastAPI server
    payload = {
        "days_before_departure": days_before_departure,
        "seats_remaining": seats_remaining,
        "demand_index": demand_index,
        "competitor_avg_price": competitor_avg_price,
        "route_popularity_score": route_popularity_score,
        "base_ticket_price": base_ticket_price,
        "total_seats": total_seats
    }
    
    try:
        # Call the local FastAPI backend
        response = requests.post("http://localhost:8000/predict_price", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            predicted_price = result["predicted_dynamic_price"]
            multiplier = result["surge_multiplier"]
            
            # Display results in metrics cards
            st.divider()
            st.subheader("Pricing Output")
            m1, m2, m3 = st.columns(3)
            
            m1.metric("Base Price", f"${base_ticket_price:,.2f}")
            m2.metric("Predicted Dynamic Price", f"${predicted_price:,.2f}", delta=f"{multiplier}x multiplier", delta_color="inverse")
            m3.metric("Competitor Price", f"${competitor_avg_price:,.2f}")
            
            if predicted_price > base_ticket_price:
                st.warning(f"Surge pricing active due to market conditions.")
            else:
                st.success(f"Discounted/Base pricing active.")
                
        else:
            st.error(f"Error from API: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Pricing API. Please ensure your FastAPI server is running on localhost:8000.")