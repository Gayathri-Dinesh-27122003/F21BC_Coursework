FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for Flask
EXPOSE 5000

# Set environment
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the web app
CMD ["python3", "app.py"]
