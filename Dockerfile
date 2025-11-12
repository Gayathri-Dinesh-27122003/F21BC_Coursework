FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set working directory to src
WORKDIR /app/src

# Run the demo
CMD ["python3", "main.py"]
