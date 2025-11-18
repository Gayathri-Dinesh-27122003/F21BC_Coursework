#!/bin/bash
# Start PSO-ANN Web Application

echo "PSO-ANN Web Application Starter"
echo "================================"
echo ""

# Check if data exists
if [ ! -f "data/processed_data.npz" ]; then
    echo "Error: Data file not found!"
    echo "Please run data preprocessing first:"
    echo "  python3 src/data_loader.ipynb"
    exit 1
fi

echo "Starting Flask application..."
echo "Open browser to: http://localhost:5000"
echo ""

python3 app.py
