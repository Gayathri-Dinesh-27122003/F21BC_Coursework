#!/bin/bash
# Startup script for Render deployment

echo "Starting PSO-ANN Web Application..."
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

# Check if data exists
if [ -f "data/processed_data.npz" ]; then
    echo "✓ Data file found"
else
    echo "✗ Warning: Data file not found!"
fi

# Start gunicorn with proper settings
exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --keep-alive 5 app:app
