from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add the project root to the system path to import your pricing engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pricing_engine.dynamic_pricing import predict_dynamic_price

app = FastAPI(
    title="Airline Dynamic Pricing API",
    description="API for predicting flight ticket prices dynamically based on ML model.",
    version="1.0"
)

# Define the expected JSON payload using Pydantic
class FlightPricingRequest(BaseModel):
    days_before_departure: int
    seats_remaining: int
    demand_index: float
    competitor_avg_price: float
    route_popularity_score: int
    base_ticket_price: float
    total_seats: int = 150

class PricingResponse(BaseModel):
    base_price: float
    predicted_dynamic_price: float
    surge_multiplier: float

@app.get("/")
def health_check():
    return {"status": "Pricing Engine is Online"}

@app.post("/predict_price", response_model=PricingResponse)
def get_price(request: FlightPricingRequest):
    try:
        # Call the function from your dynamic_pricing.py
        final_price = predict_dynamic_price(
            days_before_departure=request.days_before_departure,
            seats_remaining=request.seats_remaining,
            demand_index=request.demand_index,
            competitor_avg_price=request.competitor_avg_price,
            route_popularity_score=request.route_popularity_score,
            base_ticket_price=request.base_ticket_price,
            total_seats=request.total_seats
        )
        
        multiplier = round(final_price / request.base_ticket_price, 2)
        
        return PricingResponse(
            base_price=request.base_ticket_price,
            predicted_dynamic_price=final_price,
            surge_multiplier=multiplier
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    # Run from terminal using: python api/app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)