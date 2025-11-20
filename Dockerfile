FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py .
COPY config.json .
COPY src/ ./src/
COPY data/ ./data/
COPY static/ ./static/
COPY templates/ ./templates/

# Expose port for Flask
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Run the web app
CMD ["python", "app.py"]
