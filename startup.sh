#!/bin/bash
# Azure App Service startup script for Financial Projection Backend

echo "Starting Financial Projection Backend on Azure..."

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/home/site/wwwroot"

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI application
python -m uvicorn "BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION:app" --host 0.0.0.0 --port 8000